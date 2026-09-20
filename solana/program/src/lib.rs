//! Control-plane attestation for the Pi cluster. Inference stays on the LAN; this
//! program only records which nodes are serving, under which epoch, and who answered
//! each job. One Cluster PDA per authority key, one Job PDA per request.
//!
//! # Instructions (borsh enum, u8 tag)
//!
//! | tag | name          | accounts                                                   |
//! |-----|---------------|------------------------------------------------------------|
//! | 0   | Initialize    | authority (signer, writable), cluster PDA (writable), system |
//! | 1   | RegisterNode  | authority (signer), cluster PDA (writable)                   |
//! | 2   | SetWorkerSet  | authority (signer), cluster PDA (writable)                   |
//! | 3   | CommitJob     | authority (signer, writable), cluster PDA (writable), job PDA (writable), system |
//! | 4   | CloseJob      | authority (signer, writable), cluster PDA, job PDA (writable) |
//!
//! `CloseJob` drains a Job PDA's rent back to the authority and hands the account to
//! the system program with no data, so the per-request rent is recoverable once a job
//! no longer needs to be provable on chain. The job id can be committed again after
//! that: closing is the authority's explicit decision to forget the record.
//!
//! The account layouts of [`Cluster`] and [`Job`] and the tags 0..=3 are decoded by
//! `router/attest.py` and by a live Devnet account; they must not change. New
//! instructions get new tags appended to the enum.
//!
//! # PDA creation
//!
//! Anyone can send lamports to a PDA before it exists. `SystemInstruction::CreateAccount`
//! refuses an address that already holds lamports, which would let a stranger block
//! `Initialize` (the cluster address is derivable from the authority key) or a
//! `CommitJob` whose request id they can guess. [`create_pda`] therefore tops the
//! address up to rent exemption and uses `Allocate` + `Assign` whenever it is already
//! funded, and only uses `CreateAccount` for a fresh address.
//!
//! # Errors
//!
//! Custom error codes are the discriminants of [`AttestError`]; everything else is a
//! standard `ProgramError` (`MissingRequiredSignature`, `IncorrectProgramId`, ...).

use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    log::sol_log_64,
    msg,
    program::{invoke, invoke_signed},
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    sysvar::Sysvar,
};
use solana_system_interface::{
    instruction::{allocate, assign, create_account, transfer},
    program::ID as SYSTEM_PROGRAM_ID,
};

pub const CLUSTER_SEED: &[u8] = b"cluster";
pub const JOB_SEED: &[u8] = b"job";
pub const CLUSTER_SPACE: usize = 1024;
pub const JOB_SPACE: usize = 128;
pub const MAX_NODES: usize = 16;
pub const MAX_HOST_LEN: usize = 40;
pub const MAX_SERVED_BY_LEN: usize = 16;

pub const STATE_DOWN: u8 = 0;
pub const STATE_HEALTHY: u8 = 1;
pub const STATE_DEGRADED: u8 = 2;
pub const STATE_RESTARTING: u8 = 3;

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub host: String,
    pub active: bool,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    pub authority: Pubkey,
    pub model_hash: [u8; 32],
    pub epoch: u64,
    pub state: u8,
    pub nodes: Vec<Node>,
    pub jobs_total: u64,
    pub jobs_local: u64,
    pub last_job: [u8; 16],
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub cluster: Pubkey,
    pub job_id: [u8; 16],
    pub epoch: u64,
    pub served_by: String,
    pub result_hash: [u8; 32],
}

/// Tags are the variant order; the Python client hand-encodes 0..=3. Append only.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    /// [authority: signer+writable payer, cluster PDA: writable, system program]
    Initialize { model_hash: [u8; 32] },
    /// [authority: signer, cluster PDA: writable]
    RegisterNode { host: String },
    /// [authority: signer, cluster PDA: writable]. Bumps the epoch.
    SetWorkerSet { state: u8, active: Vec<String> },
    /// [authority: signer+writable payer, cluster PDA: writable, job PDA: writable, system program]
    CommitJob { job_id: [u8; 16], served_by: String, result_hash: [u8; 32] },
    /// [authority: signer+writable (receives the rent), cluster PDA, job PDA: writable].
    /// Drains the job account to the authority and returns it to the system program.
    CloseJob { job_id: [u8; 16] },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AttestError {
    Unauthorized = 0,
    UnknownNode = 1,
    TooManyNodes = 2,
    BadState = 3,
    WrongPda = 4,
    FieldTooLong = 5,
    NotInitialized = 6,
    /// The job account's recorded cluster or id does not match the accounts passed.
    WrongJob = 7,
    /// The PDA already holds data or is already owned by this program.
    AlreadyExists = 8,
}

impl From<AttestError> for ProgramError {
    fn from(e: AttestError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

// ----- pure state transitions (unit-tested off-chain) --------------------------

pub fn register_node(cluster: &mut Cluster, host: &str) -> Result<(), AttestError> {
    if host.is_empty() || host.len() > MAX_HOST_LEN {
        return Err(AttestError::FieldTooLong);
    }
    if cluster.nodes.iter().any(|n| n.host == host) {
        return Ok(()); // idempotent: the sidecar re-registers on every restart
    }
    if cluster.nodes.len() >= MAX_NODES {
        return Err(AttestError::TooManyNodes);
    }
    cluster.nodes.push(Node { host: host.to_string(), active: false });
    Ok(())
}

pub fn set_worker_set(cluster: &mut Cluster, state: u8, active: &[String]) -> Result<(), AttestError> {
    if state > STATE_RESTARTING {
        return Err(AttestError::BadState);
    }
    for host in active {
        if !cluster.nodes.iter().any(|n| &n.host == host) {
            return Err(AttestError::UnknownNode);
        }
    }
    for node in cluster.nodes.iter_mut() {
        node.active = active.contains(&node.host);
    }
    cluster.state = state;
    cluster.epoch += 1;
    Ok(())
}

pub fn commit_job(cluster: &mut Cluster, job_id: [u8; 16], served_by: &str) -> Result<(), AttestError> {
    if served_by.is_empty() || served_by.len() > MAX_SERVED_BY_LEN {
        return Err(AttestError::FieldTooLong);
    }
    cluster.jobs_total += 1;
    if served_by == "cluster" {
        cluster.jobs_local += 1;
    }
    cluster.last_job = job_id;
    Ok(())
}

// ----- account plumbing --------------------------------------------------------

fn load_cluster(account: &AccountInfo, program_id: &Pubkey, authority: &AccountInfo) -> Result<Cluster, ProgramError> {
    if account.owner != program_id {
        return Err(AttestError::NotInitialized.into());
    }
    if !authority.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    let cluster = Cluster::deserialize(&mut &account.data.borrow()[..])?;
    if &cluster.authority != authority.key {
        return Err(AttestError::Unauthorized.into());
    }
    Ok(cluster)
}

fn store<T: BorshSerialize>(account: &AccountInfo, value: &T) -> ProgramResult {
    let bytes = borsh::to_vec(value)?;
    let mut data = account.data.borrow_mut();
    if bytes.len() > data.len() {
        return Err(ProgramError::AccountDataTooSmall);
    }
    data[..bytes.len()].copy_from_slice(&bytes);
    Ok(())
}

/// Creates `pda` (derived from `seeds`) owned by this program with `space` bytes, paid
/// by `payer`. Works whether or not the address was funded beforehand; see the module
/// docs. Fails with `AlreadyExists` if the address already carries data or an owner.
fn create_pda<'a>(
    payer: &AccountInfo<'a>,
    pda: &AccountInfo<'a>,
    system_program: &AccountInfo<'a>,
    program_id: &Pubkey,
    seeds: &[&[u8]],
    space: usize,
) -> ProgramResult {
    let (expected, bump) = Pubkey::find_program_address(seeds, program_id);
    if pda.key != &expected {
        return Err(AttestError::WrongPda.into());
    }
    if !payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if system_program.key != &SYSTEM_PROGRAM_ID {
        msg!("expected the system program");
        return Err(ProgramError::IncorrectProgramId);
    }
    if pda.owner != &SYSTEM_PROGRAM_ID || !pda.data_is_empty() {
        return Err(AttestError::AlreadyExists.into());
    }
    let required = Rent::get()?.minimum_balance(space);
    let bump = [bump];
    let mut signer_seeds: Vec<&[u8]> = seeds.to_vec();
    signer_seeds.push(&bump);
    let signers: &[&[&[u8]]] = &[&signer_seeds];

    let funded = pda.lamports();
    if funded == 0 {
        return invoke_signed(
            &create_account(payer.key, pda.key, required, space as u64, program_id),
            &[payer.clone(), pda.clone(), system_program.clone()],
            signers,
        );
    }
    // Pre-funded address: CreateAccount would fail with AccountAlreadyInUse. Top up
    // to rent exemption from the payer, then allocate and assign with the PDA seeds.
    msg!("pda pre-funded, allocating in place");
    let shortfall = required.saturating_sub(funded);
    if shortfall > 0 {
        invoke(
            &transfer(payer.key, pda.key, shortfall),
            &[payer.clone(), pda.clone(), system_program.clone()],
        )?;
    }
    invoke_signed(
        &allocate(pda.key, space as u64),
        &[pda.clone(), system_program.clone()],
        signers,
    )?;
    invoke_signed(
        &assign(pda.key, program_id),
        &[pda.clone(), system_program.clone()],
        signers,
    )
}

/// Moves every lamport from `account` to `recipient`, zeroes and empties its data and
/// returns it to the system program. The runtime then garbage-collects it.
fn close_account<'a>(account: &AccountInfo<'a>, recipient: &AccountInfo<'a>) -> ProgramResult {
    let lamports = account.lamports();
    let new_balance = recipient
        .lamports()
        .checked_add(lamports)
        .ok_or(ProgramError::ArithmeticOverflow)?;
    **account.try_borrow_mut_lamports()? = 0;
    **recipient.try_borrow_mut_lamports()? = new_balance;
    account.try_borrow_mut_data()?.fill(0);
    account.resize(0)?;
    account.assign(&SYSTEM_PROGRAM_ID);
    Ok(())
}

#[cfg(not(feature = "no-entrypoint"))]
solana_program::entrypoint!(process_instruction);

pub fn process_instruction(program_id: &Pubkey, accounts: &[AccountInfo], data: &[u8]) -> ProgramResult {
    let ix = Instruction::try_from_slice(data).map_err(|_| ProgramError::InvalidInstructionData)?;
    let iter = &mut accounts.iter();
    let authority = next_account_info(iter)?;
    let cluster_acc = next_account_info(iter)?;

    match ix {
        Instruction::Initialize { model_hash } => {
            let system_program = next_account_info(iter)?;
            create_pda(authority, cluster_acc, system_program, program_id,
                       &[CLUSTER_SEED, authority.key.as_ref()], CLUSTER_SPACE)?;
            let cluster = Cluster {
                authority: *authority.key,
                model_hash,
                epoch: 0,
                state: STATE_DOWN,
                nodes: Vec::new(),
                jobs_total: 0,
                jobs_local: 0,
                last_job: [0; 16],
            };
            msg!("cluster initialized");
            store(cluster_acc, &cluster)
        }
        Instruction::RegisterNode { host } => {
            let mut cluster = load_cluster(cluster_acc, program_id, authority)?;
            register_node(&mut cluster, &host)?;
            msg!("node registered");
            store(cluster_acc, &cluster)
        }
        Instruction::SetWorkerSet { state, active } => {
            let mut cluster = load_cluster(cluster_acc, program_id, authority)?;
            set_worker_set(&mut cluster, state, &active)?;
            msg!("worker set: epoch, state, active");
            sol_log_64(cluster.epoch, state as u64, active.len() as u64, 0, 0);
            store(cluster_acc, &cluster)
        }
        Instruction::CommitJob { job_id, served_by, result_hash } => {
            let job_acc = next_account_info(iter)?;
            let system_program = next_account_info(iter)?;
            let mut cluster = load_cluster(cluster_acc, program_id, authority)?;
            commit_job(&mut cluster, job_id, &served_by)?;
            // the job PDA must not exist yet: a job cannot be committed twice
            create_pda(authority, job_acc, system_program, program_id,
                       &[JOB_SEED, cluster_acc.key.as_ref(), &job_id], JOB_SPACE)?;
            let job = Job { cluster: *cluster_acc.key, job_id, epoch: cluster.epoch, served_by, result_hash };
            msg!("job committed");
            store(job_acc, &job)?;
            store(cluster_acc, &cluster)
        }
        Instruction::CloseJob { job_id } => {
            let job_acc = next_account_info(iter)?;
            // proves the signer is the authority of this cluster
            load_cluster(cluster_acc, program_id, authority)?;
            let (expected, _) =
                Pubkey::find_program_address(&[JOB_SEED, cluster_acc.key.as_ref(), &job_id], program_id);
            if job_acc.key != &expected {
                return Err(AttestError::WrongPda.into());
            }
            if job_acc.owner != program_id {
                return Err(AttestError::NotInitialized.into());
            }
            let job = Job::deserialize(&mut &job_acc.data.borrow()[..])?;
            if &job.cluster != cluster_acc.key || job.job_id != job_id {
                return Err(AttestError::WrongJob.into());
            }
            close_account(job_acc, authority)?;
            msg!("job closed");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster() -> Cluster {
        Cluster {
            authority: Pubkey::new_unique(),
            model_hash: [7; 32],
            epoch: 0,
            state: STATE_DOWN,
            nodes: Vec::new(),
            jobs_total: 0,
            jobs_local: 0,
            last_job: [0; 16],
        }
    }

    #[test]
    fn register_is_idempotent_and_bounded() {
        let mut c = cluster();
        register_node(&mut c, "192.168.50.11").unwrap();
        register_node(&mut c, "192.168.50.11").unwrap();
        assert_eq!(c.nodes.len(), 1);
        for i in 0..MAX_NODES - 1 {
            register_node(&mut c, &format!("10.0.0.{i}")).unwrap();
        }
        assert_eq!(register_node(&mut c, "10.0.1.1"), Err(AttestError::TooManyNodes));
        assert_eq!(register_node(&mut c, ""), Err(AttestError::FieldTooLong));
    }

    #[test]
    fn worker_set_bumps_epoch_and_rejects_unknown_hosts() {
        let mut c = cluster();
        register_node(&mut c, "a").unwrap();
        register_node(&mut c, "b").unwrap();
        set_worker_set(&mut c, STATE_HEALTHY, &["a".into(), "b".into()]).unwrap();
        assert_eq!(c.epoch, 1);
        assert!(c.nodes.iter().all(|n| n.active));
        set_worker_set(&mut c, STATE_DEGRADED, &["a".into()]).unwrap();
        assert_eq!(c.epoch, 2);
        assert_eq!(c.nodes.iter().filter(|n| n.active).count(), 1);
        assert_eq!(set_worker_set(&mut c, STATE_HEALTHY, &["zzz".into()]), Err(AttestError::UnknownNode));
        assert_eq!(set_worker_set(&mut c, 9, &[]), Err(AttestError::BadState));
        assert_eq!(c.epoch, 2, "failed transitions do not bump the epoch");
    }

    #[test]
    fn jobs_count_local_and_cloud() {
        let mut c = cluster();
        commit_job(&mut c, [1; 16], "cluster").unwrap();
        commit_job(&mut c, [2; 16], "baseten").unwrap();
        assert_eq!((c.jobs_total, c.jobs_local, c.last_job), (2, 1, [2; 16]));
        assert_eq!(commit_job(&mut c, [3; 16], "x".repeat(17).as_str()), Err(AttestError::FieldTooLong));
    }

    #[test]
    fn full_cluster_fits_in_account() {
        let mut c = cluster();
        for i in 0..MAX_NODES {
            register_node(&mut c, &format!("192.168.100.{}", 100 + i)).unwrap();
        }
        assert!(borsh::to_vec(&c).unwrap().len() <= CLUSTER_SPACE);
        let j = Job { cluster: Pubkey::new_unique(), job_id: [1; 16], epoch: 1,
                      served_by: "x".repeat(MAX_SERVED_BY_LEN), result_hash: [2; 32] };
        assert!(borsh::to_vec(&j).unwrap().len() <= JOB_SPACE);
    }

    #[test]
    fn instruction_layout_is_stable() {
        // the Python sidecar hand-encodes these layouts and the Devnet account holds them; pin every tag
        let mut initialize = vec![0u8];
        initialize.extend_from_slice(&[9; 32]);
        assert_eq!(borsh::to_vec(&Instruction::Initialize { model_hash: [9; 32] }).unwrap(), initialize);

        let ix = Instruction::RegisterNode { host: "ab".into() };
        assert_eq!(borsh::to_vec(&ix).unwrap(), vec![1, 2, 0, 0, 0, b'a', b'b']);

        let ix = Instruction::SetWorkerSet { state: 2, active: vec!["a".into()] };
        assert_eq!(borsh::to_vec(&ix).unwrap(), vec![2, 2, 1, 0, 0, 0, 1, 0, 0, 0, b'a']);

        let ix = Instruction::CommitJob { job_id: [4; 16], served_by: "c".into(), result_hash: [5; 32] };
        let mut commit = vec![3u8];
        commit.extend_from_slice(&[4; 16]);
        commit.extend_from_slice(&[1, 0, 0, 0, b'c']);
        commit.extend_from_slice(&[5; 32]);
        assert_eq!(borsh::to_vec(&ix).unwrap(), commit);

        let mut close = vec![4u8];
        close.extend_from_slice(&[6; 16]);
        assert_eq!(borsh::to_vec(&Instruction::CloseJob { job_id: [6; 16] }).unwrap(), close);
        assert_eq!(close.len(), 17);
    }

    #[test]
    fn account_layouts_are_stable() {
        // Cluster: authority(32) model_hash(32) epoch(8) state(1) nodes(4 + n*(4+len+1)) jobs_total(8) jobs_local(8) last_job(16)
        let mut c = cluster();
        register_node(&mut c, "ab").unwrap();
        let bytes = borsh::to_vec(&c).unwrap();
        assert_eq!(bytes.len(), 32 + 32 + 8 + 1 + (4 + 4 + 2 + 1) + 8 + 8 + 16);
        assert_eq!(&bytes[..32], c.authority.as_ref());
        assert_eq!(&bytes[73..77], &[1, 0, 0, 0], "nodes vec length");
        assert_eq!(&bytes[81..84], &[b'a', b'b', 0], "host then active flag");
        // Job: cluster(32) job_id(16) epoch(8) served_by(4+len) result_hash(32)
        let j = Job { cluster: Pubkey::new_unique(), job_id: [1; 16], epoch: 3, served_by: "xy".into(), result_hash: [2; 32] };
        assert_eq!(borsh::to_vec(&j).unwrap().len(), 32 + 16 + 8 + (4 + 2) + 32);
    }

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(ProgramError::from(AttestError::Unauthorized), ProgramError::Custom(0));
        assert_eq!(ProgramError::from(AttestError::NotInitialized), ProgramError::Custom(6));
        assert_eq!(ProgramError::from(AttestError::WrongJob), ProgramError::Custom(7));
        assert_eq!(ProgramError::from(AttestError::AlreadyExists), ProgramError::Custom(8));
    }
}

/// Account-level tests: the compiled program runs through the real Agave runtime
/// (mollusk-svm). They need `target/deploy/cluster_attest.so`, produced by
/// `cargo build-sbf --arch v3` (or run everything with `cargo test-sbf --arch v3`).
#[cfg(test)]
mod svm_tests {
    use super::*;
    use mollusk_svm::{program::keyed_account_for_system_program, result::ProgramResult as SvmResult, Mollusk};
    use solana_account::Account;
    use solana_instruction::{AccountMeta, Instruction as SvmInstruction};

    const AUTHORITY_LAMPORTS: u64 = 10_000_000_000;

    fn program_elf() -> Vec<u8> {
        let candidates = std::env::var("SBF_OUT_DIR")
            .into_iter()
            .map(std::path::PathBuf::from)
            .chain(std::iter::once(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/deploy"),
            ))
            .map(|d| d.join("cluster_attest.so"));
        for path in candidates {
            if let Ok(bytes) = std::fs::read(&path) {
                return bytes;
            }
        }
        panic!("cluster_attest.so not found: run `cargo build-sbf --arch v3` first (or `cargo test-sbf --arch v3`)");
    }

    struct Env {
        mollusk: Mollusk,
        program_id: Pubkey,
        authority: Pubkey,
        cluster_pda: Pubkey,
    }

    impl Env {
        fn new() -> Self {
            let program_id = Pubkey::new_unique();
            let mut mollusk = Mollusk::default();
            mollusk.add_program_with_loader_and_elf(
                &program_id,
                &mollusk_svm::program::loader_keys::LOADER_V3,
                &program_elf(),
            );
            let authority = Pubkey::new_unique();
            let (cluster_pda, _) = Pubkey::find_program_address(&[CLUSTER_SEED, authority.as_ref()], &program_id);
            Self { mollusk, program_id, authority, cluster_pda }
        }

        fn rent(&self, space: usize) -> u64 {
            self.mollusk.sysvars.rent.minimum_balance(space)
        }

        fn wallet(&self) -> Account {
            Account { lamports: AUTHORITY_LAMPORTS, owner: SYSTEM_PROGRAM_ID, ..Account::default() }
        }

        fn job_pda(&self, job_id: [u8; 16]) -> Pubkey {
            Pubkey::find_program_address(&[JOB_SEED, self.cluster_pda.as_ref(), &job_id], &self.program_id).0
        }

        /// A program-owned account holding `value`, padded to `space`, rent exempt.
        fn owned_account<T: BorshSerialize>(&self, value: &T, space: usize) -> Account {
            let mut data = borsh::to_vec(value).unwrap();
            data.resize(space, 0);
            Account { lamports: self.rent(space), data, owner: self.program_id, ..Account::default() }
        }

        fn cluster_state(&self) -> Cluster {
            Cluster {
                authority: self.authority,
                model_hash: [7; 32],
                epoch: 0,
                state: STATE_DOWN,
                nodes: vec![Node { host: "pi1".into(), active: false }],
                jobs_total: 0,
                jobs_local: 0,
                last_job: [0; 16],
            }
        }

        fn cluster_account(&self) -> Account {
            self.owned_account(&self.cluster_state(), CLUSTER_SPACE)
        }

        fn job_account(&self, job_id: [u8; 16]) -> Account {
            let job = Job { cluster: self.cluster_pda, job_id, epoch: 1, served_by: "cluster".into(), result_hash: [3; 32] };
            self.owned_account(&job, JOB_SPACE)
        }

        fn ix(&self, ix: &Instruction, metas: Vec<AccountMeta>) -> SvmInstruction {
            SvmInstruction { program_id: self.program_id, accounts: metas, data: borsh::to_vec(ix).unwrap() }
        }

        fn initialize_ix(&self, cluster_pda: Pubkey) -> SvmInstruction {
            self.ix(&Instruction::Initialize { model_hash: [7; 32] }, vec![
                AccountMeta::new(self.authority, true),
                AccountMeta::new(cluster_pda, false),
                AccountMeta::new_readonly(SYSTEM_PROGRAM_ID, false),
            ])
        }

        fn commit_ix(&self, job_id: [u8; 16], job_pda: Pubkey) -> SvmInstruction {
            self.ix(&Instruction::CommitJob { job_id, served_by: "cluster".into(), result_hash: [3; 32] }, vec![
                AccountMeta::new(self.authority, true),
                AccountMeta::new(self.cluster_pda, false),
                AccountMeta::new(job_pda, false),
                AccountMeta::new_readonly(SYSTEM_PROGRAM_ID, false),
            ])
        }

        fn close_ix(&self, signer: Pubkey, job_id: [u8; 16]) -> SvmInstruction {
            self.ix(&Instruction::CloseJob { job_id }, vec![
                AccountMeta::new(signer, true),
                AccountMeta::new_readonly(self.cluster_pda, false),
                AccountMeta::new(self.job_pda(job_id), false),
            ])
        }

        fn run(&self, ix: &SvmInstruction, accounts: Vec<(Pubkey, Account)>) -> mollusk_svm::result::InstructionResult {
            let mut accounts = accounts;
            accounts.push(keyed_account_for_system_program());
            self.mollusk.process_instruction(ix, &accounts)
        }
    }

    fn account<'a>(result: &'a mollusk_svm::result::InstructionResult, key: &Pubkey) -> &'a Account {
        &result.resulting_accounts.iter().find(|(k, _)| k == key).expect("account in result").1
    }

    fn decode_cluster(account: &Account) -> Cluster {
        Cluster::deserialize(&mut &account.data[..]).unwrap()
    }

    #[track_caller]
    fn assert_custom(result: &mollusk_svm::result::InstructionResult, err: AttestError) {
        assert_eq!(result.program_result, SvmResult::Failure(ProgramError::Custom(err as u32)));
    }

    #[test]
    fn initialize_creates_cluster_pda() {
        let env = Env::new();
        let result = env.run(&env.initialize_ix(env.cluster_pda), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, Account::default()),
        ]);
        assert_eq!(result.program_result, SvmResult::Success);
        let pda = account(&result, &env.cluster_pda);
        assert_eq!(pda.owner, env.program_id);
        assert_eq!(pda.data.len(), CLUSTER_SPACE);
        assert_eq!(pda.lamports, env.rent(CLUSTER_SPACE));
        let cluster = decode_cluster(pda);
        assert_eq!(cluster.authority, env.authority);
        assert_eq!(cluster.model_hash, [7; 32]);
        assert_eq!(account(&result, &env.authority).lamports, AUTHORITY_LAMPORTS - env.rent(CLUSTER_SPACE));
    }

    #[test]
    fn initialize_prefunded_pda_succeeds() {
        // finding 1: a stranger sends lamports to the derivable cluster address before Initialize
        let env = Env::new();
        let rent = env.rent(CLUSTER_SPACE);
        for prefund in [1u64, rent / 2, rent, rent + 12_345] {
            let result = env.run(&env.initialize_ix(env.cluster_pda), vec![
                (env.authority, env.wallet()),
                (env.cluster_pda, Account { lamports: prefund, owner: SYSTEM_PROGRAM_ID, ..Account::default() }),
            ]);
            assert_eq!(result.program_result, SvmResult::Success, "prefund {prefund}");
            let pda = account(&result, &env.cluster_pda);
            assert_eq!(pda.owner, env.program_id);
            assert_eq!(pda.data.len(), CLUSTER_SPACE);
            assert_eq!(pda.lamports, rent.max(prefund), "topped up to rent exemption, never drained");
            assert_eq!(decode_cluster(pda).authority, env.authority);
            let paid = AUTHORITY_LAMPORTS - account(&result, &env.authority).lamports;
            assert_eq!(paid, rent.saturating_sub(prefund), "payer covers only the shortfall");
        }
    }

    #[test]
    fn commit_job_prefunded_pda_succeeds() {
        let env = Env::new();
        let job_id = [9; 16];
        let job_pda = env.job_pda(job_id);
        let result = env.run(&env.commit_ix(job_id, job_pda), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (job_pda, Account { lamports: 5, owner: SYSTEM_PROGRAM_ID, ..Account::default() }),
        ]);
        assert_eq!(result.program_result, SvmResult::Success);
        let job = account(&result, &job_pda);
        assert_eq!(job.owner, env.program_id);
        assert_eq!(Job::deserialize(&mut &job.data[..]).unwrap().job_id, job_id);
        assert_eq!(decode_cluster(account(&result, &env.cluster_pda)).jobs_total, 1);
    }

    #[test]
    fn initialize_twice_fails() {
        let env = Env::new();
        let result = env.run(&env.initialize_ix(env.cluster_pda), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_custom(&result, AttestError::AlreadyExists);
    }

    #[test]
    fn initialize_rejects_wrong_system_program() {
        let env = Env::new();
        let fake_system = Pubkey::new_unique();
        let ix = env.ix(&Instruction::Initialize { model_hash: [7; 32] }, vec![
            AccountMeta::new(env.authority, true),
            AccountMeta::new(env.cluster_pda, false),
            AccountMeta::new_readonly(fake_system, false),
        ]);
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, Account::default()),
            (fake_system, Account { executable: true, ..Account::default() }),
        ]);
        assert_eq!(result.program_result, SvmResult::Failure(ProgramError::IncorrectProgramId));
    }

    #[test]
    fn non_signer_authority_is_rejected() {
        let env = Env::new();
        let ix = env.ix(&Instruction::RegisterNode { host: "pi2".into() }, vec![
            AccountMeta::new_readonly(env.authority, false),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_eq!(result.program_result, SvmResult::Failure(ProgramError::MissingRequiredSignature));
    }

    #[test]
    fn wrong_authority_is_unauthorized() {
        let env = Env::new();
        let stranger = Pubkey::new_unique();
        let ix = env.ix(&Instruction::RegisterNode { host: "pi2".into() }, vec![
            AccountMeta::new_readonly(stranger, true),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let result = env.run(&ix, vec![
            (stranger, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_custom(&result, AttestError::Unauthorized);
    }

    #[test]
    fn system_owned_cluster_is_not_initialized() {
        let env = Env::new();
        let ix = env.ix(&Instruction::RegisterNode { host: "pi2".into() }, vec![
            AccountMeta::new_readonly(env.authority, true),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let mut unowned = env.cluster_account();
        unowned.owner = SYSTEM_PROGRAM_ID;
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, unowned),
        ]);
        assert_custom(&result, AttestError::NotInitialized);
    }

    #[test]
    fn commit_job_wrong_pda_is_rejected() {
        let env = Env::new();
        let job_id = [1; 16];
        let bogus = Pubkey::new_unique();
        let result = env.run(&env.commit_ix(job_id, bogus), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (bogus, Account::default()),
        ]);
        assert_custom(&result, AttestError::WrongPda);
    }

    #[test]
    fn commit_job_twice_fails() {
        let env = Env::new();
        let job_id = [2; 16];
        let job_pda = env.job_pda(job_id);
        let first = env.run(&env.commit_ix(job_id, job_pda), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (job_pda, Account::default()),
        ]);
        assert_eq!(first.program_result, SvmResult::Success);
        let second = env.run(&env.commit_ix(job_id, job_pda), first.resulting_accounts);
        assert_custom(&second, AttestError::AlreadyExists);
    }

    #[test]
    fn set_worker_set_rejects_unregistered_host() {
        let env = Env::new();
        let ix = env.ix(&Instruction::SetWorkerSet { state: STATE_HEALTHY, active: vec!["ghost".into()] }, vec![
            AccountMeta::new_readonly(env.authority, true),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_custom(&result, AttestError::UnknownNode);

        let ok = env.ix(&Instruction::SetWorkerSet { state: STATE_HEALTHY, active: vec!["pi1".into()] }, vec![
            AccountMeta::new_readonly(env.authority, true),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let result = env.run(&ok, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_eq!(result.program_result, SvmResult::Success);
        let cluster = decode_cluster(account(&result, &env.cluster_pda));
        assert_eq!((cluster.epoch, cluster.state), (1, STATE_HEALTHY));
        assert!(cluster.nodes[0].active);
    }

    #[test]
    fn register_node_rejects_long_host() {
        let env = Env::new();
        let ix = env.ix(&Instruction::RegisterNode { host: "h".repeat(MAX_HOST_LEN + 1) }, vec![
            AccountMeta::new_readonly(env.authority, true),
            AccountMeta::new(env.cluster_pda, false),
        ]);
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
        ]);
        assert_custom(&result, AttestError::FieldTooLong);
    }

    #[test]
    fn close_job_returns_rent_to_authority() {
        let env = Env::new();
        let job_id = [4; 16];
        let job_pda = env.job_pda(job_id);
        let result = env.run(&env.close_ix(env.authority, job_id), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (job_pda, env.job_account(job_id)),
        ]);
        assert_eq!(result.program_result, SvmResult::Success);
        let job = account(&result, &job_pda);
        assert_eq!(job.lamports, 0);
        assert!(job.data.is_empty());
        assert_eq!(job.owner, SYSTEM_PROGRAM_ID);
        assert_eq!(account(&result, &env.authority).lamports, AUTHORITY_LAMPORTS + env.rent(JOB_SPACE));
        assert_eq!(decode_cluster(account(&result, &env.cluster_pda)), env.cluster_state(), "cluster untouched");

        // the id can be committed again afterwards
        let again = env.run(&env.commit_ix(job_id, job_pda), result.resulting_accounts);
        assert_eq!(again.program_result, SvmResult::Success);
        assert_eq!(account(&again, &job_pda).owner, env.program_id);
    }

    #[test]
    fn close_job_by_stranger_fails() {
        let env = Env::new();
        let job_id = [4; 16];
        let stranger = Pubkey::new_unique();
        let result = env.run(&env.close_ix(stranger, job_id), vec![
            (stranger, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (env.job_pda(job_id), env.job_account(job_id)),
        ]);
        assert_custom(&result, AttestError::Unauthorized);
        assert_eq!(account(&result, &env.job_pda(job_id)).lamports, env.rent(JOB_SPACE), "nothing drained");
    }

    #[test]
    fn close_job_checks_pda_and_ownership() {
        let env = Env::new();
        let job_id = [4; 16];
        let other = Pubkey::new_unique();
        let ix = env.ix(&Instruction::CloseJob { job_id }, vec![
            AccountMeta::new(env.authority, true),
            AccountMeta::new_readonly(env.cluster_pda, false),
            AccountMeta::new(other, false),
        ]);
        let result = env.run(&ix, vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (other, env.job_account(job_id)),
        ]);
        assert_custom(&result, AttestError::WrongPda);

        // the right address but never created: nothing to close
        let result = env.run(&env.close_ix(env.authority, job_id), vec![
            (env.authority, env.wallet()),
            (env.cluster_pda, env.cluster_account()),
            (env.job_pda(job_id), Account::default()),
        ]);
        assert_custom(&result, AttestError::NotInitialized);
    }
}

//! Control-plane attestation for the Pi cluster. Inference stays on the LAN; this
//! program only records which nodes are serving, under which epoch, and who answered
//! each job. One Cluster PDA per authority key, one Job PDA per request.

use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program::invoke_signed,
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    sysvar::Sysvar,
};
use solana_system_interface::instruction::create_account;

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
    let lamports = Rent::get()?.minimum_balance(space);
    let bump = [bump];
    let mut signer_seeds: Vec<&[u8]> = seeds.to_vec();
    signer_seeds.push(&bump);
    invoke_signed(
        &create_account(payer.key, pda.key, lamports, space as u64, program_id),
        &[payer.clone(), pda.clone(), system_program.clone()],
        &[&signer_seeds],
    )
}

entrypoint!(process_instruction);

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
            msg!("node registered: {}", host);
            store(cluster_acc, &cluster)
        }
        Instruction::SetWorkerSet { state, active } => {
            let mut cluster = load_cluster(cluster_acc, program_id, authority)?;
            set_worker_set(&mut cluster, state, &active)?;
            msg!("epoch {} state {} active {}", cluster.epoch, state, active.len());
            store(cluster_acc, &cluster)
        }
        Instruction::CommitJob { job_id, served_by, result_hash } => {
            let job_acc = next_account_info(iter)?;
            let system_program = next_account_info(iter)?;
            let mut cluster = load_cluster(cluster_acc, program_id, authority)?;
            commit_job(&mut cluster, job_id, &served_by)?;
            // create_account fails if the job PDA exists: a job cannot be committed twice
            create_pda(authority, job_acc, system_program, program_id,
                       &[JOB_SEED, cluster_acc.key.as_ref(), &job_id], JOB_SPACE)?;
            let job = Job { cluster: *cluster_acc.key, job_id, epoch: cluster.epoch, served_by, result_hash };
            msg!("job committed by {}", job.served_by);
            store(job_acc, &job)?;
            store(cluster_acc, &cluster)
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
        // the Python sidecar hand-encodes this layout; pin it
        let ix = Instruction::SetWorkerSet { state: 2, active: vec!["a".into()] };
        assert_eq!(borsh::to_vec(&ix).unwrap(), vec![2, 2, 1, 0, 0, 0, 1, 0, 0, 0, b'a']);
    }
}

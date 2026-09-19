# Person 3 — Cluster Pair & Performance Runbook

Your deliverable is not a product surface. It is **four numbers and one timeline**, plus a cluster
that stays up. Everything below exists to produce those.

The four numbers:

| # | Number | Where it comes from | Used by |
|---|--------|--------------------|---------|
| 1 | Cluster tokens/sec, before and after the straggler fix | `loadtest.py` summary, two runs | Main pitch headline |
| 2 | Recovery time: cable pulled → tokens streaming again | `metrics.py` CSV event marks | The unplug beat |
| 3 | TTFT p50/p95 under realistic concurrency | `loadtest.py` summary | "It's actually usable" |
| 4 | % of requests served locally vs Baseten | Person 2's router log + `X-Served-By` tally | Baseten + main close |

---

## 0. Setup (Friday, do this before anything else)

### Current topology

`pi-node-1` .. `pi-node-4` are the inference nodes, `pi-node-5` runs the router.
User `pi`, password `pi`. **Currently on Wi-Fi, not the switch** — see the warning in section 0b.

Reach them by mDNS hostname rather than IP, so DHCP reshuffles don't break anything:

```bash
ping -c 2 pi-node-1.local
ssh pi@pi-node-1.local hostname -I     # what does it think its own address is
```

If `.local` doesn't resolve, the Pi needs avahi (`sudo apt install -y avahi-daemon`) or you fall
back to IPs. Find them with `arp -a` or `nmap -sn <your-subnet>/24`.

### Password auth

You're using `pi` / `pi` rather than keys. Automated tools can't type a password at a prompt, so
they need `sshpass`:

```bash
sudo apt install sshpass                          # Debian / Ubuntu / WSL
brew install hudochenkov/sshpass/sshpass          # macOS (not in the main tap)
```

Then put the password in an environment variable rather than on the command line, so it doesn't
show up in `ps` or your shell history:

```bash
export SSHPW=pi
./metrics.py --nodes pi-node-1.local,pi-node-2.local,pi-node-3.local,pi-node-4.local \
             --user pi --password-env SSHPW --csv baseline_idle.csv
```

`--password pi` also works if you don't care.

For your own ad-hoc commands you'll still be typing the password each time:

```bash
for n in 1 2 3 4; do
  echo -n "pi-node-$n: "
  sshpass -e ssh -o StrictHostKeyChecking=accept-new pi@pi-node-$n.local \
    "vcgencmd measure_temp; vcgencmd get_throttled; vcgencmd measure_clock arm" | tr '\n' ' '
  echo
done
```

(With `SSHPW` exported, use `sshpass -p "$SSHPW"` or `SSHPASS="$SSHPW" sshpass -e`.)

If the password prompts get old, one command switches you over and nothing else changes:
`ssh-copy-id pi@pi-node-1.local` for each node, then drop `--password-env`.

### 0b. The Wi-Fi problem — read this before you record any number

distributed-llama splits one model across nodes and synchronises **every layer, every token**. It is
bandwidth- and latency-bound, not compute-bound. The original plan called for ~940 Mbps wired with
sub-millisecond latency for exactly this reason.

Wi-Fi gives you a fraction of that, and worse, it's a *shared half-duplex* medium: all four nodes
contend for the same airtime, so their traffic collides with each other. Expect several times lower
throughput and jitter measured in milliseconds rather than microseconds.

What this means concretely:

- **Wi-Fi is fine for tonight's setup work.** Bootstrapping, installing deps, verifying SSH, checking
  `vcgencmd` works, testing that `metrics.py` and `loadtest.py` run. Do all of it.
- **Every performance number you take on Wi-Fi is throwaway.** Idle temp baselines survive the move
  to the switch. Tokens/sec, TTFT, and the straggler ranking do not. Don't put them in the pitch and
  don't spend Saturday tuning against them.
- **Re-baseline the moment the switch is in.** The straggler you find on Wi-Fi may just be the node
  with the worst antenna position.
- Before you measure anything real, confirm the path: `iperf3 -s` on pi-node-1 and
  `iperf3 -c pi-node-1.local` from each other node. Wired should be ~940 Mbps. Whatever Wi-Fi gives
  you, write it down so you can show the before/after.

If the switch isn't arriving, that's a scope conversation to have tonight, not Saturday afternoon.
A 2-node cluster over Wi-Fi may well beat a 4-node one, because you halve the sync traffic.

---

## 1. Capture idle baselines BEFORE any load

```bash
export SSHPW=pi
./metrics.py --csv baseline_idle.csv --user pi --password-env SSHPW
# defaults to pi-node-1..4.local; let it run 3 minutes idle, then Ctrl-C
```

Write down each node's idle temp and idle ARM clock. You need these to recognise abnormal later.

**Read the throttle flags now, not during the demo.** `vcgencmd get_throttled` should be `0x0` at
idle on a healthy node.

| Bits | Meaning | Reaction |
|------|---------|----------|
| `0x1` / `0x2` / `0x4` / `0x8` | under-voltage NOW / freq capped NOW / **throttled NOW** / soft temp limit NOW | live problem, act |
| `0x10000`–`0x80000` | the same four things **have happened since boot** | history, sticky until reboot |

The plan's note that `0x50000` means "currently throttled" is off by a category: `0x50000` is bits 16
and 18, meaning *under-voltage has occurred* and *throttling has occurred* at some point since boot.
The live bit is `0x4`. `metrics.py` splits these: live flags print in red and **shout**, historical
flags print dim and lowercase. Chase the red ones.

**If a node shows under-voltage at idle, it is a power problem, not a heat problem.** This is the
single most common Pi-cluster straggler and no amount of fan repositioning fixes it. Swap the USB-C
cable first (thin cables drop enough voltage to trigger it), then the PSU. Four Pi 5s under full
CPU load will brown out a cheap 4-port charger — each wants its own supply.

---

## 2. Tools

### `metrics.py` — the demo dashboard

One persistent SSH connection per node running a small remote loop. No per-poll connection setup,
and a node losing power shows `down` within ~6 seconds instead of hanging a poll.

```bash
./metrics.py \
  --nodes pi-node-1.local,pi-node-2.local,pi-node-3.local,pi-node-4.local \
  --user pi --password-env SSHPW \
  --csv run_$(date +%H%M).csv \
  --status-url http://pi-node-1.local:9991/status \
  --tps-file /tmp/dllama_tps.json
```

- `--status-url` polls Person 1's supervisor JSON, so `healthy → restarting → degraded` shows up
  on the same screen as the node going dark. That one line is most of the failure story.
- `--tps-file` reads the sidecar `loadtest.py` writes, giving live client-measured tokens/sec.
  Prefer this over log scraping; it is what the user actually experiences.
- `--log-path` + `--tps-regex` will tail the root node's log instead, if you'd rather. Check the
  real log format before relying on it, and use `--tps-unit ms_per_token` if it reports latency.
- **Press ENTER to drop an event marker into the CSV.** Hit it the instant you pull a cable. That
  marker is how you turn "it recovered pretty fast" into "it recovered in 11 seconds".
- The straggler line under the table shows hottest node, coolest node, and the spread. A spread over
  ~6 °C turns magenta. That's your hunt, visible without reading numbers.

### `loadtest.py` — the load generator

```bash
# sanity: does it stream at all
./loadtest.py --url http://pi-node-1.local:9990/v1/chat/completions -c 1 -n 3

# burst: latency under concurrency
./loadtest.py --url ... -c 4 -n 40 --prompt-tokens 300 --max-tokens 150 --out burst.jsonl

# the 15-minute sustained run (straggler hunt)
./loadtest.py --url ... -c 4 --duration 900 \
  --prompt-tokens 400 --max-tokens 200 \
  --out sustained_before.jsonl --tps-file /tmp/dllama_tps.json --label before-fix

# through the router, to prove escalation fires
./loadtest.py --url http://pi-node-5.local:8000/v1/chat/completions -c 2 -n 10 --prompt-tokens 4000
```

Prompts are shaped like coding-agent traffic (shell one-liners, refactors, short Q&A) so the numbers
transfer to the Warp demo. `--think-time` adds a pause between requests if you want a gentler
long-burn profile. The summary tallies the `X-Served-By` header, which is Person 2's routing
evidence and yours in one output.

---

## 3. The straggler hunt (Saturday afternoon)

Two terminals side by side. Left: `metrics.py`. Right: `loadtest.py --duration 900`.

1. Run 15 minutes. Watch for the node whose temp climbs fastest and whose ARM clock starts dropping
   below the others. On a Pi 5 that's a fall from ~2400 MHz; on a Pi 4, from ~1800 MHz.
2. When the clock drops on one node and aggregate tok/s sags with it, you've found it.
3. **Change one thing, rerun, record.** Candidates, in the order worth trying:
   - power/cable (if any under-voltage bit is live — always fix this first)
   - physical position: the hot node is usually the one in the middle of the stack with no airflow
   - heatsink seating / fan direction
   - `--nthreads 3` instead of 4 on that node, leaving a core for the OS and network stack
   - worker ordering in Person 1's launch command
4. Rerun with `--label after-fix` and a fresh `--out`. The two summary blocks side by side are the
   pitch stat. Screenshot both.

Do not change two things at once. You will not know which one worked and you will have no story.

---

## 4. Failure-drill protocol (validate the unplug with data)

1. Start `metrics.py --csv drill1.csv --status-url ...`.
2. Start `loadtest.py -c 2 --duration 180 --tps-file /tmp/dllama_tps.json`.
3. At ~t+60s: press ENTER in the metrics terminal, type `pull-node3`, then pull node3's power.
4. Watch: node3 → `down`, supervisor → `restarting` → `degraded`, client tok/s dips then recovers.
   Note whether `loadtest.py` records any failed requests. Zero errors means the router covered the
   gap, which is the whole point.
5. Press ENTER again with `restored`, plug node3 back in, and time the return to 4 nodes.
6. Repeat five times. Report the median, not the best one.

Extract the recovery time from the CSV:

```bash
python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open('drill1.csv')))
marks = [(float(r['ts_unix']), r['event']) for r in rows if r['event']]
print(marks)
# then find the first row after the pull mark where cluster_status returns to healthy
EOF
```

---

## 5. Saturday night long burn

An hour of moderate load (`-c 2 --duration 3600 --think-time 1 --out longburn.jsonl`) while everyone
else polishes. Leave `metrics.py` recording. In the morning, check: any `down` states, any error rows
in the JSONL, any memory trending downward in `mem_avail_mb` (that's a leak, and it will surface on
stage if you ignore it). File anything you find straight to Person 1 or 2 that night.

A crash tonight is a gift. The same crash Sunday is a disaster.

---

## 6. Sunday

You run `metrics.py` during every rehearsal and the real demo, and you are the hands on the cable.
Person 4 talks, you pull. One person doing both fumbles.

Pre-demo checklist:
- [ ] metrics terminal open, font size large enough to read from the judges' distance
- [ ] `--no-color` off, window sized so the table doesn't wrap
- [ ] CSV filename for the real run set before you start
- [ ] SSH connections already warm (start metrics 2 minutes early)
- [ ] baseline and after-fix numbers on a sticky note, in case someone asks mid-demo

---

## Gotchas worth knowing before they cost you two hours

- **Wi-Fi shadow route.** Once the switch is in, if the Pis still have `wlan0` up on the same
  subnet, inter-node traffic can silently keep taking the slow path and your throughput will be
  mysteriously unchanged. After cabling: `sudo rfkill block wifi` on all four, or make sure the
  route metric strongly prefers `eth0`. Then re-run iperf3 to confirm you actually moved.
- **Power-save on the Wi-Fi chip** adds tens of milliseconds of latency at random. While you're
  still wireless: `sudo iw dev wlan0 set power_save off` on every node.
- **`nthreads 4` on a 4-core Pi** leaves nothing for the OS and the network stack. If TTFT is jittery
  rather than slow, try 3.
- **Sticky throttle bits never clear** until reboot. If you want to know whether *tonight's* run
  throttled, reboot the node first or watch only the live bits.
- **SD card throughput** matters at model load time, not during inference. A slow card shows up as a
  long startup, not low tok/s. Don't chase it.
- **Your laptop is a variable.** Run `loadtest.py` from the same machine every time, on ethernet if
  possible. A client on congested venue Wi-Fi will report TTFT numbers that have nothing to do with
  the cluster.
- **Timestamps.** `metrics.py` stamps locally, so your CSV and your loadtest JSONL always line up
  even if the Pis' clocks drift. Node-side log lines are the ones that need NTP.

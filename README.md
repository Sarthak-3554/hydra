# Fault-Tolerant Distributed File Replication (C++ / MPI)

A cluster-node replication system inspired by HDFS/GFS-style block replication and BitTorrent-style
swarming — built in C++ with MPI as the transport layer. Every piece of a file is deterministically
placed on K nodes, self-heals when a node goes silent, and the whole thing runs on a fixed set of
MPI processes with zero external dependencies.

This started as a BitTorrent-style file downloader and evolved into a fault-tolerant replicated
storage prototype as the project scope shifted from "share a file" to "keep data available even
when machines fail."

## What it actually does

1. Rank 0 (the "seeder") reads a file and splits it into fixed-size pieces.
2. Every other rank ("storage node") is deterministically assigned ownership of a subset of pieces,
   such that every piece ends up on exactly **K** nodes (configurable replication factor).
3. Ownership is computed by a formula every node runs independently — no coordinator, no races,
   no messages exchanged just to decide who fetches what.
4. Nodes periodically heartbeat each other. If a node goes silent, the rest of the swarm detects it,
   drops it from the ownership rotation, and automatically re-replicates its pieces elsewhere to
   restore the target replication factor.
5. On completion, the program reports exactly how well-replicated the data ended up, and — if a
   failure was simulated — how long recovery took.

## Real-world parallels

| This project | Real system |
|---|---|
| Replication factor K, piece-based storage | HDFS block replication (default factor 3), GFS chunkservers |
| Deterministic piece ownership by formula | Consistent hashing (Cassandra), CRUSH placement (Ceph) |
| Heartbeat-based failure detection | Cassandra's gossip protocol, HDFS DataNode heartbeats |
| MPI-based internal cluster distribution | Facebook's internal BitTorrent-based deploy tool, Twitter's "Murder" |

## Build

```bash
# macOS
brew install open-mpi

# Ubuntu / WSL
sudo apt install openmpi-bin libopenmpi-dev

mpic++ -O2 -o mpi_torrent mpi_torrent.cpp
```

## Run

Normal operation (no simulated failure):
```bash
head -c 2097152 /dev/urandom > input.txt   # or use a real file
mpirun --oversubscribe -np 6 ./mpi_torrent
```

With a simulated node failure (rank 2 goes silent after holding a few pieces):
```bash
mpirun --oversubscribe -np 6 ./mpi_torrent 2
```

Check the end of the output for:
```
[Replication Report] Target K=3 | Min=3 | Max=3 | Avg=3 | Exactly at K=32/32 | Over-replicated=0/32 | TARGET MET
```
and, if a failure was simulated:
```
[Recovery] Full replication restored 0.0066 seconds after failure was detected.
```

## Verifying correctness

Each node writes `output_peer_<rank>.bin` containing only the pieces it holds. Under replication,
not every node holds the full file — that's by design. To verify a specific node's partial data is
byte-correct against the source, check which piece IDs it logged receiving and compare those exact
byte ranges against the original file (see `verify.py` if included, or reconstruct manually using
the piece size and IDs from the log).

## Architecture

- **Transport**: MPI point-to-point messages (`MPI_Send`/`MPI_Irecv`) for piece transfer, plus three
  lightweight gossip channels: `HAVE_UPDATE` (who has what), `FAILURE_DETECTED` (who's unreachable),
  and `HEARTBEAT` (periodic liveness ping).
- **Ownership**: `is_assigned_owner(piece_id, rank, size, K, alive)` — a pure function every node
  evaluates independently. Because it's recomputed over the *currently alive* leecher set, a node's
  failure automatically shifts its responsibilities onto the remaining nodes without any explicit
  "rebalance" step.
- **Termination**: `MPI_Allreduce` with `MPI_LAND` every 10 loop iterations — every rank votes
  "are all pieces sufficiently replicated by my knowledge," and the job exits only when everyone
  agrees. The target dynamically caps down if too many nodes have failed for the original K to be
  achievable.

## Design decisions worth knowing (and why they changed)

**Piece selection went through three iterations before landing on the final approach:**

1. *Greedy "grab anything missing"* (original) — every node pulled every piece, no replication
   target, no fault tolerance.
2. *Rarest-first with a claim-broadcast* — nodes announced "I'm fetching X" before requesting, to
   avoid two nodes grabbing the same under-replicated piece. This worked in low-concurrency testing
   but was fundamentally a **timing-dependent mitigation**: under real multi-core concurrency, nodes
   reached the decision point in near-lockstep faster than the broadcast could arrive, so it still
   over-replicated (measured avg 5.2 replicas against a target of 3).
3. *Deterministic ownership* (final) — every node computes the same answer via a formula, with zero
   coordination. This eliminates the race by construction rather than trying to outrun it. Verified
   to hit exactly K replicas per piece, 0% over-replication, at every concurrency level tested
   (6, 10, 16 processes).

**Failure detection also went through two iterations:**

1. *Request-timeout only* — if a fetch to a peer hung too long, declare it dead. This missed the
   actual failure case entirely: if no node happened to need data *from* the failed peer, nothing
   ever timed out, and the system livelocked forever waiting for replication that would never
   complete.
2. *Heartbeat protocol* (final) — nodes ping each other independent of data transfer, so a silent
   node is detected regardless of whether anyone was talking to it. This is the same principle
   behind Cassandra's gossip-based failure detector.

## Known limitations (honest scope)

- **Standard MPI has no built-in fault tolerance.** A real OS-level process crash brings down the
  entire `mpirun` job, not just that rank. "Failure" here is simulated by a rank going silent
  (stops serving/fetching) while the process itself stays alive only to keep participating in
  required MPI collective calls — a faithful simulation from the swarm's perspective, but not a
  literal process kill.
- **Fixed node set at launch.** MPI requires all processes to be known via `mpirun -np N` (or a
  hostfile) at startup. Nodes can fail during a run, but new nodes can't join dynamically — there's
  no peer-discovery mechanism.
- **No persistence across restarts.** If the whole job exits, replicated data only exists in each
  rank's memory during that run; nothing is durably written until final output.
- **No security model.** No encryption in transit, no piece integrity hashing, no peer
  authentication. Fine for a trusted internal cluster; not suitable as-is for untrusted networks.
- **Seeder (rank 0) failure isn't simulated.** The current failure simulation targets a leecher
  rank; extending it to the seeder would need the file source itself to be replicated from the
  start rather than living on a single designated rank.

## Possible next steps

- Rack-aware placement using `MPI_Get_processor_name()` so replicas are guaranteed to land on
  different physical machines, not just different ranks.
- Multi-machine deployment via an MPI hostfile, so nodes run across real, separate hosts instead of
  one machine.
- Per-piece integrity hashing (SHA-256) verified on receipt.
- Persist replicated data to disk so it survives a full job restart.

## Tested configuration

- OpenMPI 5.0.9
- macOS (Apple Silicon) and Ubuntu 24.04 (WSL2)
- Verified at 6, 10, and 16 MPI processes with a 2 MB test file (32 pieces at 64 KB each)

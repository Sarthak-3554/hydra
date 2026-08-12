#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <climits>
using namespace std;

#define PIECE_SIZE    65536
#define MAX_PIECES    20000
#define REQUEST_PIECE 1
#define SEND_PIECE    2
#define HAVE_UPDATE   3
#define FAILURE_DETECTED 4     // gossip: "peer X is unreachable, stop counting on it"
#define HEARTBEAT     5        // periodic "I'm still alive" ping
#define REPLICATION_FACTOR 3   // target number of copies per piece
#define REQUEST_TIMEOUT_ITERS 50   // loops to wait for a direct reply before suspecting the source
#define HEARTBEAT_TIMEOUT_ITERS 100 // loops without ANY heartbeat before declaring a peer dead
#define FAIL_AFTER_PIECES 3       // simulated-failure rank goes silent after holding this many

struct Piece {
    int  id;
    int  actual_size;
    char data[PIECE_SIZE];
};

// ── File I/O ───────────────────────────────────────────────────────────────
void split_file(const char* filename, vector<Piece>& pieces) {
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "ERROR: cannot open input file\n"; return; }
    int id = 0;
    while (!file.eof() && id < MAX_PIECES) {
        Piece p;
        memset(p.data, 0, PIECE_SIZE);
        p.id = id;
        file.read(p.data, PIECE_SIZE);
        int got = (int)file.gcount();
        if (got <= 0) break;
        p.actual_size = got;
        pieces.push_back(p);
        id++;
    }
}

void write_file(vector<Piece>& pieces, const char* output) {
    sort(pieces.begin(), pieces.end(),
         [](const Piece& a, const Piece& b){ return a.id < b.id; });
    ofstream file(output, ios::binary);
    for (auto& p : pieces)
        file.write(p.data, p.actual_size);
}

// ── Tell all peers I now have this piece ──────────────────────────────────
void announce_have(int piece_id, int rank, int size) {
    for (int p = 0; p < size; p++) {
        if (p == rank) continue;
        MPI_Send(&piece_id, 1, MPI_INT, p, HAVE_UPDATE, MPI_COMM_WORLD);
    }
}

// ── Read all waiting HAVE_UPDATE messages ─────────────────────────────────
void drain_have_updates(vector<vector<int>>& peer_have) {
    while (true) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, HAVE_UPDATE, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;
        int piece_id;
        MPI_Recv(&piece_id, 1, MPI_INT,
                 status.MPI_SOURCE, HAVE_UPDATE, MPI_COMM_WORLD, &status);
        peer_have[status.MPI_SOURCE][piece_id] = 1;
    }
}

// ── Tell all peers "rank X is unreachable" so everyone's view converges ───
void announce_failure(int dead_rank, int rank, int size) {
    for (int p = 0; p < size; p++) {
        if (p == rank) continue;
        MPI_Send(&dead_rank, 1, MPI_INT, p, FAILURE_DETECTED, MPI_COMM_WORLD);
    }
}

// ── Read all waiting FAILURE_DETECTED messages ────────────────────────────
void drain_failure_updates(vector<int>& alive) {
    while (true) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, FAILURE_DETECTED, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;
        int dead_rank;
        MPI_Recv(&dead_rank, 1, MPI_INT,
                 status.MPI_SOURCE, FAILURE_DETECTED, MPI_COMM_WORLD, &status);
        alive[dead_rank] = 0;
    }
}

// ── Periodic "I'm still alive" ping — this is what makes failure detection
// work regardless of whether anyone happens to be transferring data with
// the failed peer at the time. Same idea as a gossip/heartbeat protocol in
// a real distributed system (e.g. Cassandra's failure detector).
void broadcast_heartbeat(int rank, int size) {
    for (int p = 0; p < size; p++) {
        if (p == rank) continue;
        MPI_Send(&rank, 1, MPI_INT, p, HEARTBEAT, MPI_COMM_WORLD);
    }
}

// ── Read all waiting heartbeats, recording when we last heard from each peer
void drain_heartbeats(vector<int>& last_heartbeat_iter, int current_iter) {
    while (true) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, HEARTBEAT, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;
        int sender;
        MPI_Recv(&sender, 1, MPI_INT,
                 status.MPI_SOURCE, HEARTBEAT, MPI_COMM_WORLD, &status);
        last_heartbeat_iter[sender] = current_iter;
    }
}

// ── Serve all waiting piece requests ──────────────────────────────────────
void serve_pending_requests(vector<Piece>& pieces, int rank) {
    while (true) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, REQUEST_PIECE, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;

        int piece_id;
        MPI_Recv(&piece_id, 1, MPI_INT,
                 status.MPI_SOURCE, REQUEST_PIECE, MPI_COMM_WORLD, &status);

        bool found = false;
        for (auto& p : pieces) {
            if (p.id == piece_id) {
                MPI_Send(&p, sizeof(Piece), MPI_BYTE,
                         status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
                cout << "[Peer " << rank << "] Served piece " << piece_id
                     << " to peer " << status.MPI_SOURCE << "\n";
                found = true;
                break;
            }
        }
        if (!found) {
            // Must always reply or requester hangs
            Piece dummy; memset(&dummy, 0, sizeof(Piece));
            dummy.id = piece_id; dummy.actual_size = 0;
            MPI_Send(&dummy, sizeof(Piece), MPI_BYTE,
                     status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
        }
    }
}

// ── Pick best peer to ask (prefer non-seeder to spread load), skipping
//    any peer we already believe is dead ────────────────────────────────
int pick_peer(int piece_id, int my_rank, int size,
              vector<vector<int>>& peer_have, vector<int>& alive) {
    for (int p = 1; p < size; p++) {
        if (p == my_rank) continue;
        if (!alive[p]) continue;
        if (peer_have[p][piece_id]) return p;
    }
    if (alive[0] && peer_have[0][piece_id]) return 0;
    return -1;
}

// ── How many still-ALIVE peers (by our current local knowledge) hold this
//    piece. A copy sitting on a dead node no longer counts — it can't be
//    served, so from the swarm's perspective it isn't really "replicated."
int replica_count(int piece_id, int size, vector<vector<int>>& peer_have, vector<int>& alive) {
    int count = 0;
    for (int p = 0; p < size; p++)
        if (alive[p] && peer_have[p][piece_id]) count++;
    return count;
}

// ── Deterministic piece ownership, recomputed over only the currently-alive
// leechers. When a leecher is marked dead, it drops out of everyone's
// rotation calculation identically (same gossip, same formula) — so its
// ownership responsibilities are automatically picked up by the remaining
// alive leechers, with no coordinator and no extra messages needed.
bool is_assigned_owner(int piece_id, int rank, int size, int K, vector<int>& alive) {
    if (rank == 0) return false;                 // seeder doesn't "fetch"
    if (!alive[rank]) return false;               // dead nodes don't act

    vector<int> alive_leechers;
    for (int p = 1; p < size; p++)
        if (alive[p]) alive_leechers.push_back(p);
    if (alive_leechers.empty()) return false;

    int needed = min(K - 1, (int)alive_leechers.size());
    for (int i = 0; i < needed; i++) {
        int owner = alive_leechers[(piece_id + i) % alive_leechers.size()];
        if (owner == rank) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) cerr << "Need at least 2 MPI processes.\n";
        MPI_Finalize(); return 1;
    }

    // Optional: ./mpi_torrent <fail_rank>  — that rank simulates a crash
    // partway through by going silent (stops serving + stops fetching).
    int fail_rank = (argc > 1) ? atoi(argv[1]) : -1;
    bool i_have_failed = false;

    vector<Piece>       pieces;
    vector<int>         have(MAX_PIECES, 0);
    vector<vector<int>> peer_have(size, vector<int>(MAX_PIECES, 0));
    vector<int>         alive(size, 1);   // local, gossip-updated view of who's reachable
    vector<int>         last_heartbeat_iter(size, 0);  // loop_counter when we last heard from each peer
    int total_pieces = 0;

    // Rank 0 = seeder
    if (rank == 0) {
        cout << "[Seeder] Reading file...\n";
        split_file("input.txt", pieces);
        for (auto& p : pieces) {
            have[p.id]         = 1;
            peer_have[0][p.id] = 1;
        }
        total_pieces = (int)pieces.size();
    }

    // Everyone learns the piece count
    MPI_Bcast(&total_pieces, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_pieces == 0) {
        if (rank == 0) cerr << "No pieces found.\n";
        MPI_Finalize(); return 1;
    }

    // Seeder announces all pieces so leechers know where to start
    if (rank == 0) {
        for (int i = 0; i < total_pieces; i++)
            announce_have(i, rank, size);
    }

    // Effective replication target — can't ask for more copies than we have nodes
    int K = min(REPLICATION_FACTOR, size);
    if (rank == 0)
        cout << "[Seeder] Target replication factor K=" << K
             << " (requested " << REPLICATION_FACTOR << ", capped by " << size
             << " nodes)\n";

    // Non-blocking recv state. recv_buf is allocated fresh per request
    // (never reused) so a timed-out request can be safely abandoned —
    // MPI_Cancel on a large in-flight receive is unreliable in practice
    // (it can't stop a sender mid-rendezvous-handshake for that transfer),
    // so instead of fighting that, we just orphan the old buffer/request
    // and let the next request use entirely fresh memory. The process
    // exits soon after anyway, so the leaked handle is harmless here.
    bool        waiting_for_piece = false;
    int         waiting_for_id    = -1;
    int         waiting_from_peer = -1;
    int         wait_iters        = 0;
    MPI_Request pending_recv      = MPI_REQUEST_NULL;
    Piece*      recv_buf          = nullptr;

    // Recovery-time metric: rank 0 times the gap between first learning of
    // a failure and the swarm restoring full replication afterward — this
    // is the number worth putting on a resume ("recovered in X seconds").
    double failure_detected_time = -1;
    bool   recovery_timer_running = false;

    // ── MAIN LOOP ──────────────────────────────────────────────────────────
    // Termination: every N iterations we do an MPI_Allreduce to check
    // if ALL processes have all pieces. When everyone is done, all exit
    // together cleanly. No DONE messages needed.
    int loop_counter = 0;

    while (true) {

        // Step 1: refresh who-has-what, and who's-unreachable
        drain_have_updates(peer_have);
        drain_failure_updates(alive);
        drain_heartbeats(last_heartbeat_iter, loop_counter);
        if (!i_have_failed) broadcast_heartbeat(rank, size);   // a failed node stops pinging

        // Step 1b: anyone we haven't heard a heartbeat from in too long is
        // presumed dead — this is what catches the case where nobody
        // happens to be actively fetching FROM the failed peer, so a
        // request-timeout would never fire on its own.
        for (int p = 0; p < size; p++) {
            if (p == rank || !alive[p]) continue;
            if (loop_counter - last_heartbeat_iter[p] > HEARTBEAT_TIMEOUT_ITERS) {
                cout << "[Peer " << rank << "] No heartbeat from peer " << p
                     << " in " << HEARTBEAT_TIMEOUT_ITERS << " loops — marking it unreachable\n";
                alive[p] = 0;
                announce_failure(p, rank, size);
            }
        }

        // Simulated failure trigger: once this rank has done a bit of real
        // work (so the demo shows partial replication before it happens),
        // it goes silent — stops serving and stops fetching. It still has
        // to call the MPI collectives below so the rest of the job doesn't
        // hang (vanilla MPI has no built-in fault tolerance — a real
        // process crash brings down the whole run, not just that rank).
        if (rank == fail_rank && !i_have_failed && (int)pieces.size() >= FAIL_AFTER_PIECES) {
            i_have_failed = true;
            cout << "\n[Peer " << rank << "] *** SIMULATING FAILURE — going silent now ***\n\n";
        }

        // Step 2: serve anyone requesting pieces from us (skip if we've failed)
        if (!i_have_failed)
            serve_pending_requests(pieces, rank);

        // Step 3: check if our pending non-blocking recv completed
        if (waiting_for_piece) {
            int flag = 0;
            MPI_Status status;
            MPI_Test(&pending_recv, &flag, &status);
            if (flag) {
                wait_iters = 0;
                waiting_for_piece = false;
                if (recv_buf->actual_size > 0) {
                    pieces.push_back(*recv_buf);
                    have[waiting_for_id]            = 1;
                    peer_have[rank][waiting_for_id] = 1;
                    cout << "[Peer " << rank << "] Got piece " << waiting_for_id
                         << " from peer " << waiting_from_peer << "\n";
                    announce_have(waiting_for_id, rank, size);
                } else {
                    // Peer didn't have it — clear and retry from someone else
                    peer_have[waiting_from_peer][waiting_for_id] = 0;
                }
                delete recv_buf;
                recv_buf = nullptr;
                waiting_for_id    = -1;
                waiting_from_peer = -1;
            } else {
                // No reply yet — count how long we've been waiting. If it's
                // gone on too long, the source is unreachable: treat it as
                // a failure and tell everyone. We deliberately do NOT call
                // MPI_Cancel here (unreliable for large in-flight receives
                // in practice) — we just abandon this buffer/request and
                // move on with a completely fresh one next time.
                wait_iters++;
                if (wait_iters > REQUEST_TIMEOUT_ITERS) {
                    cout << "[Peer " << rank << "] Timeout waiting on peer "
                         << waiting_from_peer << " for piece " << waiting_for_id
                         << " — marking it unreachable\n";
                    alive[waiting_from_peer] = 0;
                    announce_failure(waiting_from_peer, rank, size);
                    waiting_for_piece = false;   // recv_buf/pending_recv deliberately abandoned, not freed
                    waiting_for_id    = -1;
                    waiting_from_peer = -1;
                    wait_iters        = 0;
                }
            }
        }

        // Step 4: fetch the next piece I'm deterministically assigned to own,
        // that I don't hold yet — recomputed over currently-alive nodes, so
        // if my assignment shifted because someone else died, I pick that up
        // automatically. A failed rank does none of this — it's gone silent.
        if (!waiting_for_piece && rank != 0 && !i_have_failed) {
            for (int pid = 0; pid < total_pieces; pid++) {
                if (have[pid]) continue;
                if (!is_assigned_owner(pid, rank, size, K, alive)) continue;
                int source = pick_peer(pid, rank, size, peer_have, alive);
                if (source == -1) continue;   // nobody alive has announced it yet, try again next loop

                cout << "[Peer " << rank << "] Fetching assigned piece " << pid
                     << " from peer " << source << "\n";

                MPI_Send(&pid, 1, MPI_INT, source, REQUEST_PIECE, MPI_COMM_WORLD);
                recv_buf = new Piece();
                MPI_Irecv(recv_buf, sizeof(Piece), MPI_BYTE,
                          source, SEND_PIECE, MPI_COMM_WORLD, &pending_recv);

                waiting_for_piece = true;
                waiting_for_id    = pid;
                waiting_from_peer = source;
                wait_iters        = 0;
                break;
            }
        }

        // Step 5: check termination every 10 iterations using Allreduce
        // ---------------------------------------------------------------
        // MPI_Allreduce works like a group vote:
        //   - Each process contributes 1 (I have all pieces) or 0 (I don't)
        //   - MPI_LAND = logical AND across all processes
        //   - Result is 1 only if EVERY process voted 1
        //   - All processes get the same result simultaneously
        // This replaces the broken DONE message system entirely.
        // We check every 10 loops (not every loop) to avoid the Allreduce
        // sync cost slowing down the actual transfers.
        loop_counter++;
        if (loop_counter % 10 == 0) {
            // Rank 0 starts the recovery clock the first time it locally
            // learns the simulated-failure rank is down (via gossip) —
            // this timestamps when the swarm becomes aware it has damage
            // to repair, not when the failure itself was injected.
            if (rank == 0 && fail_rank != -1 && !alive[fail_rank] && !recovery_timer_running) {
                failure_detected_time = MPI_Wtime();
                recovery_timer_running = true;
                cout << "[Seeder] Detected peer " << fail_rank
                     << " is unreachable — recovery clock started\n";
            }

            // Am I done? "Done" now means every piece has reached the
            // target among currently-ALIVE nodes — if enough nodes have
            // died that K copies is no longer achievable, the target caps
            // down instead of waiting forever for something impossible.
            int num_alive_leechers = 0;
            for (int p = 1; p < size; p++) if (alive[p]) num_alive_leechers++;
            int required = min(K, 1 + num_alive_leechers);

            int i_am_done = 1;
            for (int pid = 0; pid < total_pieces; pid++) {
                if (replica_count(pid, size, peer_have, alive) < required) { i_am_done = 0; break; }
            }

            int everyone_done = 0;
            MPI_Allreduce(&i_am_done,     // what I contribute
                          &everyone_done, // where result goes
                          1,              // one integer
                          MPI_INT,
                          MPI_LAND,       // logical AND — true only if all are 1
                          MPI_COMM_WORLD);

            if (everyone_done) {
                cout << "[Peer " << rank << "] All peers done, exiting.\n";
                if (rank == 0 && recovery_timer_running) {
                    double recovery_seconds = MPI_Wtime() - failure_detected_time;
                    cout << "\n[Recovery] Full replication restored "
                         << recovery_seconds << " seconds after failure was detected.\n";
                }
                break; // everyone exits the loop at the same time
            }
        }
    }

    // Any abandoned (timed-out) receive is deliberately left alone here —
    // see the note in Step 3 on why we don't attempt to cancel it.

    MPI_Barrier(MPI_COMM_WORLD);

    // Only nodes that actually ended up holding pieces write an output —
    // under replication, not every node holds every piece anymore.
    if (rank != 0 && !pieces.empty()) {
        string filename = "output_peer_" + to_string(rank) + ".bin";
        write_file(pieces, filename.c_str());
        string status = i_have_failed ? " [FAILED — stale data, excluded from swarm]" : "";
        cout << "[Peer " << rank << "] Wrote " << filename
             << " (" << pieces.size() << "/" << total_pieces << " pieces held)" << status << "\n";
    }

    // Rank 0 prints a final replication report — this is the number worth
    // putting on a resume/demo: did every piece actually reach K copies?
    if (rank == 0) {
        int min_replicas = INT_MAX, max_replicas = 0;
        int sum_replicas = 0, exactly_at_K = 0, over_K = 0;
        for (int pid = 0; pid < total_pieces; pid++) {
            int rc = replica_count(pid, size, peer_have, alive);
            min_replicas = min(min_replicas, rc);
            max_replicas = max(max_replicas, rc);
            sum_replicas += rc;
            if (rc == K) exactly_at_K++;
            if (rc > K)  over_K++;
        }
        double avg = (double)sum_replicas / total_pieces;
        cout << "\n[Replication Report] Target K=" << K
             << " | Min=" << min_replicas
             << " | Max=" << max_replicas
             << " | Avg=" << avg
             << " | Exactly at K=" << exactly_at_K << "/" << total_pieces
             << " | Over-replicated=" << over_K << "/" << total_pieces
             << " | " << (min_replicas >= min(K, size) ? "TARGET MET" : "UNDER-REPLICATED") << "\n";
        if (fail_rank != -1)
            cout << "[Replication Report] Simulated failure of peer " << fail_rank
                 << " — its data excluded from the counts above (correctly treated as unreachable)\n";
    }

    MPI_Finalize();
    return 0;
}
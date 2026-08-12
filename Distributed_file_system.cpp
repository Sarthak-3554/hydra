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
#define REPLICATION_FACTOR 3   // target number of copies per piece

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

// ── Pick best peer to ask (prefer non-seeder to spread load) ──────────────
int pick_peer(int piece_id, int my_rank, int size,
              vector<vector<int>>& peer_have) {
    for (int p = 1; p < size; p++) {
        if (p == my_rank) continue;
        if (peer_have[p][piece_id]) return p;
    }
    if (peer_have[0][piece_id]) return 0;
    return -1;
}

// ── How many peers (by our current local knowledge) hold this piece ───────
int replica_count(int piece_id, int size, vector<vector<int>>& peer_have) {
    int count = 0;
    for (int p = 0; p < size; p++)
        if (peer_have[p][piece_id]) count++;
    return count;
}

// ── Deterministic piece ownership — the real fix for the race condition.
// Instead of nodes RACING to decide who fetches an under-replicated piece
// (which a claim-broadcast can only ever reduce, never eliminate, under
// real concurrency), every node runs the SAME formula and independently
// arrives at the SAME answer for "who owns this piece" — with zero
// messages exchanged. There is nothing left to race over.
//
// Rank 0 (seeder) already holds everything and counts as 1 of the K copies.
// The remaining (K-1) copies are assigned to leechers by rotating through
// them based on piece id, so ownership spreads evenly instead of piling
// onto the first few ranks.
bool is_assigned_owner(int piece_id, int rank, int size, int K) {
    if (rank == 0) return false;              // seeder doesn't "fetch"
    int num_leechers = size - 1;               // ranks 1..size-1
    int needed = min(K - 1, num_leechers);      // copies needed beyond the seeder
    for (int i = 0; i < needed; i++) {
        int owner = 1 + ((piece_id + i) % num_leechers);
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

    vector<Piece>       pieces;
    vector<int>         have(MAX_PIECES, 0);
    vector<vector<int>> peer_have(size, vector<int>(MAX_PIECES, 0));
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

    // Non-blocking recv state
    bool        waiting_for_piece = false;
    int         waiting_for_id    = -1;
    int         waiting_from_peer = -1;
    MPI_Request pending_recv      = MPI_REQUEST_NULL;
    Piece       recv_buf;

    // ── MAIN LOOP ──────────────────────────────────────────────────────────
    // Termination: every N iterations we do an MPI_Allreduce to check
    // if ALL processes have all pieces. When everyone is done, all exit
    // together cleanly. No DONE messages needed.
    int loop_counter = 0;

    while (true) {

        // Step 1: refresh who-has-what
        drain_have_updates(peer_have);

        // Step 2: serve anyone requesting pieces from us
        serve_pending_requests(pieces, rank);

        // Step 3: check if our pending non-blocking recv completed
        if (waiting_for_piece) {
            int flag = 0;
            MPI_Status status;
            MPI_Test(&pending_recv, &flag, &status);
            if (flag) {
                waiting_for_piece = false;
                if (recv_buf.actual_size > 0) {
                    pieces.push_back(recv_buf);
                    have[waiting_for_id]            = 1;
                    peer_have[rank][waiting_for_id] = 1;
                    cout << "[Peer " << rank << "] Got piece " << waiting_for_id
                         << " from peer " << waiting_from_peer << "\n";
                    announce_have(waiting_for_id, rank, size);
                } else {
                    // Peer didn't have it — clear and retry from someone else
                    peer_have[waiting_from_peer][waiting_for_id] = 0;
                }
                waiting_for_id    = -1;
                waiting_from_peer = -1;
            }
        }

        // Step 4: fetch the next piece I'm deterministically assigned to own,
        // that I don't hold yet. No claim/race needed at all — every node
        // computes the same ownership answer independently, so two nodes
        // can never both decide to fetch the same piece by coincidence.
        if (!waiting_for_piece && rank != 0) {
            for (int pid = 0; pid < total_pieces; pid++) {
                if (have[pid]) continue;
                if (!is_assigned_owner(pid, rank, size, K)) continue;
                int source = pick_peer(pid, rank, size, peer_have);
                if (source == -1) continue;   // nobody has announced it yet, try again next loop

                cout << "[Peer " << rank << "] Fetching assigned piece " << pid
                     << " from peer " << source << "\n";

                MPI_Send(&pid, 1, MPI_INT, source, REQUEST_PIECE, MPI_COMM_WORLD);
                MPI_Irecv(&recv_buf, sizeof(Piece), MPI_BYTE,
                          source, SEND_PIECE, MPI_COMM_WORLD, &pending_recv);

                waiting_for_piece = true;
                waiting_for_id    = pid;
                waiting_from_peer = source;
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
            // Am I done? Under replication, "done" means every piece has
            // reached the target K copies — by MY current knowledge of
            // peer_have. Once HAVE_UPDATE broadcasts finish propagating,
            // every rank's local view converges and the Allreduce agrees.
            int i_am_done = 1;
            for (int pid = 0; pid < total_pieces; pid++) {
                if (replica_count(pid, size, peer_have) < K) { i_am_done = 0; break; }
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
                break; // everyone exits the loop at the same time
            }
        }
    }

    // Cancel any leftover pending recv
    if (waiting_for_piece) {
        MPI_Cancel(&pending_recv);
        MPI_Request_free(&pending_recv);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Only nodes that actually ended up holding pieces write an output —
    // under replication, not every node holds every piece anymore.
    if (rank != 0 && !pieces.empty()) {
        string filename = "output_peer_" + to_string(rank) + ".bin";
        write_file(pieces, filename.c_str());
        cout << "[Peer " << rank << "] Wrote " << filename
             << " (" << pieces.size() << "/" << total_pieces << " pieces held)\n";
    }

    // Rank 0 prints a final replication report — this is the number worth
    // putting on a resume/demo: did every piece actually reach K copies?
    if (rank == 0) {
        int min_replicas = INT_MAX, max_replicas = 0;
        int sum_replicas = 0, exactly_at_K = 0, over_K = 0;
        for (int pid = 0; pid < total_pieces; pid++) {
            int rc = replica_count(pid, size, peer_have);
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
             << " | " << (min_replicas >= K ? "TARGET MET" : "UNDER-REPLICATED") << "\n";
    }

    MPI_Finalize();
    return 0;
}
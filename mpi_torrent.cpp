#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
using namespace std;

#define PIECE_SIZE    256
#define MAX_PIECES    64
#define REQUEST_PIECE 1
#define SEND_PIECE    2
#define HAVE_UPDATE   3

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

        // Step 4: request next missing piece (only one at a time)
        if (!waiting_for_piece && rank != 0) {
            for (int pid = 0; pid < total_pieces; pid++) {
                if (have[pid]) continue;
                int source = pick_peer(pid, rank, size, peer_have);
                if (source == -1) continue;

                cout << "[Peer " << rank << "] Requesting piece " << pid
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
            // Am I done? Seeder always has everything. Leechers check have[].
            int i_am_done = 1;
            if (rank != 0) {
                for (int pid = 0; pid < total_pieces; pid++) {
                    if (!have[pid]) { i_am_done = 0; break; }
                }
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

    if (rank != 0 && !pieces.empty()) {
        string filename = "output_peer_" + to_string(rank) + ".bin";
        write_file(pieces, filename.c_str());
        cout << "[Peer " << rank << "] Wrote " << filename << "\n";
    }

    MPI_Finalize();
    return 0;
}

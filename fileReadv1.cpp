#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include<bits/stdc++.h>
using namespace std;

#define PIECE_SIZE   256
#define MAX_PIECES   64
#define REQUEST_PIECE 1
#define SEND_PIECE    2
#define NO_MORE_REQS  3   // signal: seeder tells a peer "I have nothing more for you"

// FIX 7: Track actual bytes in each piece so the last piece isn't padded with garbage
struct Piece {
    int  id;
    int  actual_size;          // how many bytes are valid in data[]
    char data[PIECE_SIZE];
};

// ── Split file into pieces ─────────────────────────────────────────────────
void split_file(const char* filename, vector<Piece>& pieces) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "ERROR: cannot open input file '" << filename << "'\n";
        return;
    }
    int id = 0;
    while (!file.eof() && id < MAX_PIECES) {
        Piece p;
        memset(p.data, 0, PIECE_SIZE);
        p.id = id;
        file.read(p.data, PIECE_SIZE);
        int got = (int)file.gcount();
        if (got <= 0) break;
        p.actual_size = got;   // FIX 7: store real byte count
        pieces.push_back(p);
        id++;
    }
}

// ── Write reconstructed file ───────────────────────────────────────────────
// FIX 7: write only actual_size bytes per piece, not always PIECE_SIZE
void write_file(vector<Piece>& pieces, const char* output) {
    // Sort pieces by id before writing
    sort(pieces.begin(), pieces.end(),
         [](const Piece& a, const Piece& b){ return a.id < b.id; });

    ofstream file(output, ios::binary);
    for (auto& p : pieces)
        file.write(p.data, p.actual_size);  // FIX 7
}

// ── Seeder: handle ALL pending requests in one call ───────────────────────
// FIX 3 & 2: loop until no more requests are queued, not just one
void serve_all_pending_requests(vector<Piece>& pieces) {
    while (true) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, REQUEST_PIECE, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;   // nothing left to serve right now

        int piece_id;
        MPI_Recv(&piece_id, 1, MPI_INT,
                 status.MPI_SOURCE, REQUEST_PIECE, MPI_COMM_WORLD, &status);

        bool found = false;
        for (auto& p : pieces) {
            if (p.id == piece_id) {
                MPI_Send(&p, sizeof(Piece), MPI_BYTE,
                         status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
                cout << "[Seeder] Sent piece " << piece_id
                     << " to peer " << status.MPI_SOURCE << "\n";
                found = true;
                break;
            }
        }
        if (!found) {
            // FIX 3: must always send a reply or the requester blocks forever
            // Send a dummy piece with actual_size = 0 to signal "not found"
            Piece dummy;
            memset(&dummy, 0, sizeof(Piece));
            dummy.id          = piece_id;
            dummy.actual_size = 0;
            MPI_Send(&dummy, sizeof(Piece), MPI_BYTE,
                     status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            cerr << "Need at least 2 MPI processes.\n";
        MPI_Finalize();
        return 1;
    }

    vector<Piece> pieces;
    vector<int>   have(MAX_PIECES, 0);

    // Rank 1 is the seeder
    int total_pieces = 0;
    if (rank == 1) {
        cout << "Peer 1 is seeder, reading file\n";
        split_file("input.txt", pieces);
        for (auto& p : pieces)
            have[p.id] = 1;
        total_pieces = (int)pieces.size();
    }

    // FIX 5: broadcast the real piece count from seeder to everyone
    MPI_Bcast(&total_pieces, 1, MPI_INT, 1, MPI_COMM_WORLD);

    if (total_pieces == 0) {
        if (rank == 0) cerr << "No pieces to distribute.\n";
        MPI_Finalize();
        return 1;
    }

    // ── Main download loop ────────────────────────────────────────────────
    // FIX 3 & 4: separate roles so there's no deadlock.
    //   - Rank 1 (seeder) only serves requests; it already has everything.
    //   - All other peers request pieces one at a time, blocking on the reply.
    //     Between requests the seeder keeps draining its queue.

    if (rank == 1) {
        // Seeder: serve until all peers have all pieces
        // We know each of the (size-2) non-seeder, non-rank-0 peers
        // will send exactly total_pieces requests.
        // (rank 0 is excluded per original design)
        int expected = (size - 2) * total_pieces;  // peers that actually download
        // If only 2 processes exist, no leechers — just exit
        for (int served = 0; served < expected; ) {
            MPI_Status status;
            int piece_id;
            // FIX 3: blocking recv — no busy-spin, no missed replies
            MPI_Recv(&piece_id, 1, MPI_INT,
                     MPI_ANY_SOURCE, REQUEST_PIECE, MPI_COMM_WORLD, &status);
            served++;

            bool found = false;
            for (auto& p : pieces) {
                if (p.id == piece_id) {
                    MPI_Send(&p, sizeof(Piece), MPI_BYTE,
                             status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
                    cout << "[Seeder] Sent piece " << piece_id
                         << " to peer " << status.MPI_SOURCE << "\n";
                    found = true;
                    break;
                }
            }
            if (!found) {
                Piece dummy; memset(&dummy, 0, sizeof(Piece));
                dummy.id = piece_id; dummy.actual_size = 0;
                MPI_Send(&dummy, sizeof(Piece), MPI_BYTE,
                         status.MPI_SOURCE, SEND_PIECE, MPI_COMM_WORLD);
            }
        }

    } else if (rank != 0) {
        // Leecher peers (rank >= 2): request every piece from the seeder
        for (int piece_id = 0; piece_id < total_pieces; piece_id++) {
            if (have[piece_id]) continue;

            // FIX 4: send request then do a BLOCKING recv — guaranteed reply
            MPI_Send(&piece_id, 1, MPI_INT, 1, REQUEST_PIECE, MPI_COMM_WORLD);

            Piece recv_piece;
            MPI_Status status;
            MPI_Recv(&recv_piece, sizeof(Piece), MPI_BYTE,
                     1, SEND_PIECE, MPI_COMM_WORLD, &status);  // FIX 4

            if (recv_piece.actual_size > 0) {
                pieces.push_back(recv_piece);
                have[piece_id] = 1;
                cout << "Peer " << rank << " received piece "
                     << piece_id << " from seeder\n";
            } else {
                cout << "Peer " << rank << " could not get piece "
                     << piece_id << " (not available)\n";
            }
        }
    }
    // rank 0 does nothing — matches original design

    MPI_Barrier(MPI_COMM_WORLD);

    // FIX 6: only write file for peers that actually received data
    if (rank != 0 && rank != 1 && !pieces.empty()) {
        string filename = "output_peer_" + to_string(rank) + ".bin";
        write_file(pieces, filename.c_str());
        cout << "Peer " << rank << " reconstructed file -> " << filename << "\n";
    }
    if (rank == 1) {
        // Seeder can also write its own copy
        write_file(pieces, "output_peer_1.bin");
        cout << "Peer 1 (seeder) wrote its copy\n";
    }

    MPI_Finalize();
    return 0;
}

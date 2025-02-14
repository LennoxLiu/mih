#ifndef __MIHASHER_H
#define __MIHASHER_H

#ifdef _WIN32
#include <io.h>     // For _chsize_s and access()
#include <windows.h>
#include <direct.h> // Needed for Windows file operations
#define ftruncate(fd, length) _chsize_s(fd, length)
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "types.h"
#include "bitops.h"

#include "sparse_hashtable.h"
#include "bitarray.h"

#define STAT_DIM 6		/* Dimensionality of stats, it has STAT_DIM many fields */

struct qstat {
    UINT32 numres;		// Total number of returned results
    UINT32 numcand;		// Number of hamming distance computations executed
    UINT32 numdups;		// Number of candidates skipped because they were duplicates
    UINT32 numlookups;
    UINT32 maxrho;		// Largest distance that was searched exhaustively
    clock_t ticks;		// Number of clock ticks spent on each query
};

class mihasher {
 private:

    int B;			// Bits per code

    int B_over_8;

    int b;			// Bits per chunk (must be less than 64)

    int m;			// Number of chunks

    int mplus;			// Number of chunks with b bits (have 1 bit more than others)

    int D;			// Maximum hamming search radius (we use B/2 by default)

    int d;			// Maximum hamming search radius per substring

    int K;			// Maximum results to return

    UINT64 N;			// Number of codes
	
    UINT8 *codes;		// Table of original full-length codes

    /* is not thread safe */
    bitarray *counter;		// Counter for eliminating duplicate results
	
    SparseHashtable *H;		// Array of m hashtables;
		
    UINT32 *xornum;		// Volume of a b-bit Hamming ball with radius s (for s = 0 to d)

    int power[100];		// Used within generation of binary codes at a certain Hamming distance

 public:
	
    mihasher();

    ~mihasher();

    mihasher(int B, int m);

    void setK(int K);

    void populate(UINT8 *codes, UINT32 N, int dim1codes);

    void batchquery (UINT32 *results, UINT32 *numres, qstat *stats, UINT8 * q, UINT32 numq, int dim1queries);
    
    UINT32 rangequery_single(UINT8 *query, int range_threshold, int dim1query);
 private:
    void query(UINT32 *results, UINT32* numres, qstat *stats, UINT8 *q, UINT64 * chunks, UINT32 * res);
};

#endif

/********************************************************************
 * FILE: utils.h
 * AUTHOR: William Stafford Noble
 * CREATE DATE: 9-8-97
 * PROJECT: shared
 * COPYRIGHT: 1997-2001, Columbia University
 * DESCRIPTION: Various useful generic utilities.
 ********************************************************************/
#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef linux
#include <ieeefp.h>
#endif

#define SIZE_RDNA 9100
#define NUM_RDNA_TO_ADD 148
#define RDNA_POSITION 451000

#define CENTRO_RADIUS 0.1
#define CENTRO_CENTER -0.9
//#define NUCLEO_RADIUS 0.6
//#define NUCLEO_CENTER 0.4
//#define NUCLEO_RADIUS 0.5
//#define NUCLEO_CENTER 0.5
//
#define NUCLEO_RADIUS 0.3
#define NUCLEO_CENTER 0.7

// #define NUCLEO_RADIUS 0.4
// #define NUCLEO_CENTER 0.6

#define NUM_CHROMS 30

#define WINDOW 10000
// how much to look left and right of the centromere
// 2*MARGIN is how much to look left OR right of the telomere
// not used right now

#define FALSE 0
#define TRUE 1

#define BINARY 0
#define FREQ 1
#define COUNT 2

typedef short BOOLEAN_T;

typedef int VERBOSE_T;

#define INVALID_VERBOSE -1
#define QUIET_VERBOSE 0
#define LOW_VERBOSE 1
#define NORMAL_VERBOSE 2
#define HIGH_VERBOSE 3
#define HIGHER_VERBOSE 4
#define DUMP_VERBOSE 5

extern VERBOSE_T verbosity;

/* Some Sun systems don't have these defined. */
//extern int  getopt(int, char* const *, const char* );
//extern char* optarg;
//extern int  opterr, optind, optopt;

/***********************************************************************
 * Return the value to replace the missing value.
 ***********************************************************************/
double NaN
  (void);

/********************************************************************
 * double myclock
 *
 * Return number of CPU microseconds since first call to myclock().
 * This corrects the bug in the system version of clock that causes it
 * to loop after about 36 minutes.
 *
 * (Taken from Tim Bailey's MEME package.)
 ********************************************************************/
double myclock(void);

/************************************************************************
 * int open_file
 *
 * Open a file gracefully.
 *
 * RETURN: Was the open successful?
 ************************************************************************/
BOOLEAN_T open_file
  (char*     filename,            /* Name of the file to be opened. */
   char*     file_mode,           /* Mode to be passed to fopen. */
   BOOLEAN_T allow_stdin,         /* If true, filename "-" is stdin. */
   char*     file_description,
   char*     content_description,
   FILE **         afile);              /* Pointer to the open file. */

/********************************************************************
 * DEBUG_CODE (macro)
 *
 * Allow debugging code to be included or excluded from a compiled
 * program.
 ********************************************************************/
#ifdef DEBUG
#define DEBUG_CODE( debug_value, code_fragment ) \
   { if (debug_value) { code_fragment } }
#else
#define DEBUG_CODE( debug_value, code_fragment )
#endif

/********************************************************************
 * void die()
 *
 * Print an error message and die. The arguments are formatted exactly
 * like arguments to printf().
 *
 * (Taken from Sean Eddy's HMMER package.)
 ********************************************************************/
void die
  (char* format,
   ...);

/**************************************************************************
 * Make an assertion, and print the given message if the assertion fails.
 *
 * If the first parameter is set to TRUE, then die if the assertion
 * doesn't go through.  Otherwise, just issue the warning.
 *
 * On exit, dump core if DEBUG is defined.
 **************************************************************************/
void myassert
  (BOOLEAN_T die_on_error,
   BOOLEAN_T test,
   char*  const    format,
   ...);

/********************************************************************
 * void mymalloc
 *
 * Allocate dynamic memory. Die gracefully if memory is exhausted.
 ********************************************************************/
void *mymalloc
  (size_t size);
void *mycalloc
  (size_t nelem,
   size_t size);
void * myrealloc
  (void * ptr,
   size_t size);

/********************************************************************
 * myfree (macro)
 *
 * Only free memory if the given pointer is non-null.
 ********************************************************************/
#define myfree(x) if (x) {free((char* ) (x)); (x) = NULL;}
//#define myfree(x) if (x) {free((x)); (x) = NULL;}

/********************************************************************
 * Set the seed for the random number generator.
 ********************************************************************/
void my_srand
  (long seed);

/********************************************************************
 * Get a random number X such that 0 <= X < 1.
 ********************************************************************/
double my_drand
  (void);

/********************************************************************
 * Math macros.
 ********************************************************************/
/* Note that the following type must be the  same as the MTYPE and ATYPE
   defined in 'matrix.h' and 'array.h'. */
typedef double PROB_T;        /* Type definition for probability/frequency. */
#define PROB_SCAN " %lf"      /* Scanf string for PROB_T. */

#define LOG_ZERO  (-1.0E10)  /* Zero on the log scale. */
#define LOG_SMALL (-0.5E10)  /* Threshold below which everything is zero. */
#define BITS      (33.2)     /* = LOG2(-LOG_ZERO) */

/**************************************************************************
 * Compute the logarithm of x, when 0 <= x <= 1.
 **************************************************************************/
void init_log_prob
  (void);

PROB_T log_prob
  (PROB_T value);

#define log_prob2(x)   (log_prob(x) * 1.44269504)

/**************************************************************************
 * Compute the logarithm of x.  Returns LOG_ZERO if x==0.
 **************************************************************************/
PROB_T my_log
  (PROB_T x);

#define my_log2(x)   (my_log(x) * 1.44269504)

#define my_log10(x)  (my_log(x) * 0.43429448)

#define EXP2(x)                          \
( ( (x) < LOG_SMALL) ?                   \
  0.0 :                                  \
  (exp((x) * 0.69314718 ))               \
)

/**************************************************************************
 * Given the logs (in base 2) of two numbers, return the log of their
 * sum.
 *
 * This function is optimized based upon the following formula:
 *
 *      log(x+y) = log(x) + log(1 + exp(log(y) - log(x)))
 *
 **************************************************************************/
#define LOG_VALUE(logx) \
( ( (logx) < LOG_SMALL ) ? \
    LOG_ZERO : \
    (logx) \
)

#define LOG_SUM1(logx, logy) \
( \
  ( ( (logx) - (logy) ) > BITS ) ? \
    LOG_VALUE(logx) : \
    (logx) + my_log2( 1 + EXP2((logy) - (logx) ) ) \
)

#define LOG_SUM(logx, logy) \
( \
  ( (logx) > (logy) ) ? \
    LOG_SUM1( (logx), (logy) ) : \
    LOG_SUM1( (logy), (logx) ) \
)

/**************************************************************************
 * Test for zero on a value that may be either a log or a raw float.
 **************************************************************************/
BOOLEAN_T is_zero
  (double    value,
   BOOLEAN_T log_form);

/**************************************************************************
 * Test to see if two values are approximately equal.
 **************************************************************************/
BOOLEAN_T almost_equal
  (double value1,
   double value2,
   double slop);

/*************************************************************************
 * Convert a boolean to and from a "true" or "false" string.
 *************************************************************************/
char*  boolean_to_string
 (BOOLEAN_T the_boolean);

BOOLEAN_T boolean_from_string
  (char* true_or_false);

/**************************************************************************
 * Does a given character appear in a given string?
 **************************************************************************/
BOOLEAN_T char_in_string
  (const char* a_string,
   char        a_char);

/**************************************************************************
 * Generic functions for converting between integer and string
 * representations of an enumerated type.
 *
 * Assumes that the longest string representation of the enumerated
 * type does not exceed 100 characters.
 *
 * Assumes that the zeroth enumerated type element is invalid.
 **************************************************************************/
char*  convert_enum_type
  (int     enum_type,  /* The enumerated type object to be converted. */
   char*  enum_strs[],  /* String values associated with this type. */
   int     num_enums); /* Number of values of the type. */

int convert_enum_type_str
  (char*   enum_type_str, /* String to be converted. */
   int     default_value, /* Value to return if first arg is null. */
   char**  enum_strs,     /* String values associated with this type. */
   int     num_enums);    /* Number of values of the type. */

/****************************************************************************
 * Get the name of the CPU.
 ****************************************************************************/
const char* hostname
  ();

/****************************************************************************
 * Get the current date and time.
 ****************************************************************************/
const char* date_and_time
  ();

/****************************************************************************
 * Copy a string, with allocation.
 ****************************************************************************/
char*  copy_string
 (char**  target,
  char*   source);

/*****************************************************************************
 * Randomly return an index between 0 and a given value (minus 1).
 * The function iterates through the entire range of possible values
 * before re-shuffling and iterating again.
 *****************************************************************************/
int select_random_item
  (int   num_items,
   int*  current_item,
   int** data);

#endif


/********************************************************************
 * FILE: utils.c
 * AUTHOR: William Stafford Noble
 * CREATE DATE: 9-8-97
 * PROJECT: shared
 * COPYRIGHT: 1997-2001 Columbia University
 * DESCRIPTION: Various useful generic utilities.
 ********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include <errno.h>

#ifndef VERBOSITY
#define VERBOSITY
VERBOSE_T verbosity;
#endif


/***********************************************************************
 * Return the value to replace a missing value -- NaN.
 ***********************************************************************/
double NaN
  (void)
{
  return sqrt(-1.0);
}

/**********************************************************************
 * See .h file for description.
 **********************************************************************/
#ifdef NOCLOCK
double myclock() {return(0);}
#else

#ifdef crayc90
/* No problem on the CRAY. */
#include <time.h>
double myclock() {return((double)clock());}

#else
int getrusage(int who, struct rusage *rusage);

double myclock()
{
  static BOOLEAN_T first_time = TRUE;
  static double    start_time;
  double           elapsed;
  struct rusage    ru;

  if (first_time) {
    getrusage(RUSAGE_SELF, &ru);
    start_time = (ru.ru_utime.tv_sec * 1.0E6) + ru.ru_utime.tv_usec;
    first_time = FALSE;
    return 0;

  } else {
    getrusage(RUSAGE_SELF, &ru);
    elapsed = (ru.ru_utime.tv_sec * 1.0E6) + ru.ru_utime.tv_usec -
      start_time;
    return elapsed;
  }
}
#endif /* crayc90 */
#endif /* NOCLOCK */

/************************************************************************
 * See .h file for description.
 ************************************************************************/
BOOLEAN_T open_file
  (char *    filename,            /* Name of the file to be opened. */
   char *    file_mode,           /* Mode to be passed to fopen. */
   BOOLEAN_T allow_stdin,         /* If true, filename "-" is stdin. */
   char *    file_description,
   char *    content_description,
   FILE **         afile)               /* Pointer to the open file. */
{
  errno = 0;
  if (filename == NULL) {
    fprintf(stderr, "Error: No %s filename specified.\n", file_description);
    return(FALSE);
  } else if ((allow_stdin) && (strcmp(filename, "-") == 0)) {
    if (verbosity >= HIGH_VERBOSE)
      fprintf(stderr, "Opening %s for reading\n", filename);

    if (strchr(file_mode, 'r') != NULL) {
      fprintf(stderr, "Reading %s from stdin.\n", content_description);
      *afile = stdin;
    } else if (strchr(file_mode, 'w') != NULL) {
      fprintf(stderr, "Writing %s to stdout.\n", content_description);
      *afile = stdout;
    } else {
      fprintf(stderr, "Sorry, I can't figure out whether to use stdin ");
      fprintf(stderr, "or stdout for %s.\n", content_description);
      return(FALSE);
    }
  } else if ((*afile = fopen(filename, file_mode)) == NULL) {
    fprintf(stderr, "Error opening file %s: %s.\n", filename, strerror(errno));
    return(FALSE);
  }
  return(TRUE);
}

/********************************************************************
 * See .h file for description.
 ********************************************************************/
void die
  (char *format,
   ...)
{
  va_list  argp;

  fprintf(stderr, "FATAL: ");
  va_start(argp, format);
  vfprintf(stderr, format, argp);
  va_end(argp);
  fprintf(stderr, "\n");
  fflush(stderr);

#ifdef DEBUG
  abort();
#else
  exit(1);
#endif
}


/**************************************************************************
 * See .h file for description.
 **************************************************************************/
void myassert
  (BOOLEAN_T die_on_error,
   BOOLEAN_T test,
   char * const    format,
   ...)
{
  va_list  argp;

  if (!test) {

    if (die_on_error) {
      fprintf(stderr, "FATAL: ");
    } else {
      fprintf(stderr, "WARNING: ");
    }

    /* Issue the error message. */
    va_start(argp, format);
    vfprintf(stderr, format, argp);
    va_end(argp);
    fprintf(stderr, "\n");
    fflush(stderr);

    if (die_on_error) {
#ifdef DEBUG
      abort();
#else
      exit(1);
#endif
    }
  }
}




/********************************************************************
 * void mymalloc, mycalloc, myrealloc
 *
 * See .h file for descriptions.
 ********************************************************************/
void *mymalloc
  (size_t size)
{
  void * temp_ptr;

  if (size == 0)
    size++;

  temp_ptr = malloc(size);

  if (temp_ptr == NULL) {
    die("Memory exhausted.  Cannot allocate %d bytes.", (int)size);
  } else {
#ifdef MEM_DEBUG
    fprintf(stderr, "Allocating %d bytes.\n", (int)size);
#endif
  }

  return(temp_ptr);
}

void *mycalloc
  (size_t nelem,
   size_t size)
{
  void * temp_ptr;

  /* Make sure we allocate something. */
  if (size == 0) {
    size = 1;
  }
  if (nelem == 0) {
    fprintf(stderr, "Warning: Requesting 0 items in mycalloc.\n");
    nelem = 1;
  }

  temp_ptr = calloc(nelem, size);

  if (temp_ptr == NULL) {
    die("Memory exhausted.  Cannot allocate %d bytes.", (int)size * nelem);
#ifdef MEM_DEBUG
  } else {
    fprintf(stderr, "Allocating %d bytes.\n", (int)size * nelem);
#endif
  }

  return(temp_ptr);
}

void * myrealloc
  (void * ptr,
   size_t  size)
{
  void * temp_ptr;

  /* Make sure we allocate something. */
  if (size == 0)
    size = 1;
  assert(size > 0);

  /* Some non-ANSI systems complain about reallocating NULL pointers. */
  if (ptr == NULL) {
    temp_ptr = malloc(size);
  } else {
    temp_ptr = realloc(ptr, size);
  }

  if (temp_ptr == NULL) {
    die("Memory exhausted.  Cannot allocate %d bytes.", (int)size);
#ifdef MEM_DEBUG
  } else {
    fprintf(stderr, "Re-allocating %d bytes.\n", (int)size);
#endif
  }

  return(temp_ptr);
}

#ifdef MYRAND
#define MY_RAND_MAX 4096

/********************************************************************
 * Primary function for the built-in random number generator.
 ********************************************************************/
static double my_rand
  (long seed)
{
  static long stored_seed = 0;

  /* If this is the first call, just set the seed. */
  if (stored_seed == 0) {
    stored_seed = seed;
  }

  /* Otherwise, create a new pseudorandom number. */
  else {
    stored_seed = abs((stored_seed / 3) * stored_seed + 7718);
  }

  /* Make sure the pseudorandom number is in the right range. */
  return((double)(stored_seed % MY_RAND_MAX) / (double)MY_RAND_MAX);
}
#else
/* The stupid include file doesn't have these prototypes. */
void srand48();
double drand48();

#endif

/********************************************************************
 * See .h file for description.
 ********************************************************************/
void my_srand
  (long seed)
{
#ifdef MYRAND
  my_rand(seed);
#else
  srand48(seed);
#endif
}

/********************************************************************
 * See .h file for description.
 ********************************************************************/
double my_drand
  (void)
{
#ifdef MYRAND
  return(my_rand(0));
#else
  return(drand48());
#endif
}

/**********************************************************************
 * Compute a logarithm.
 **********************************************************************/
PROB_T my_log
  (PROB_T x)
{
  if (x > 0.0) {
    return(LOG_VALUE(log(x)));
  } else if (x < 0.0) {
    die("Tried to take the log of a negative value (%g).", x);
  } /* else if (x == 0.0) */
  return(LOG_ZERO);
}

/* The lookup table. */
#define LOG_PRECISION 1.0e5
static PROB_T log_table[(int) LOG_PRECISION + 2];

/**********************************************************************
 * Set up lookup table for log(x), 0 <= x <= 1.
 **********************************************************************/
void init_log_prob
  (void)
{
  int    i_table;
  PROB_T table_value;

  log_table[0] = LOG_ZERO;
  for (i_table = 1; i_table <= LOG_PRECISION; i_table++) {
    table_value = (double)(i_table / LOG_PRECISION);
    log_table[i_table] = log(table_value);
    /*fprintf(stderr, "%d %f\n", i, log_table[i]);*/
  }
  log_table[i_table] = 0;  /* For use in iterpolation when x=1 */
}

/**********************************************************************
 * Efficiently find log(x), when 0 < x <= 1.  Doesn't check bounds.
 **********************************************************************/
PROB_T log_prob
  (PROB_T value)
{
  const PROB_T scaled_value = value * LOG_PRECISION;
  const int    log_index = (int)scaled_value;
  const PROB_T decimal_part = scaled_value - log_index;
  const PROB_T lower_value = log_table[log_index];
  const PROB_T upper_value = log_table[log_index+1];
  const PROB_T interpolation = decimal_part * (lower_value - upper_value);

  if (value == 0.0) {
    return(LOG_ZERO);
  }
  return(lower_value + interpolation);
}


/**************************************************************************
 * See .h file for description.
 **************************************************************************/
BOOLEAN_T is_zero
  (double    value,
   BOOLEAN_T log_form)
{
  if ((log_form) && (value < LOG_SMALL)) {
    return(TRUE);
  } else if ((!log_form) && (value == 0.0)) {
    return(TRUE);
  } else {
    return(FALSE);
  }
}

/**************************************************************************
 * See .h file for description.
 **************************************************************************/
BOOLEAN_T almost_equal
  (double value1,
   double value2,
   double slop)
{
  if ((value1 - slop > value2) || (value1 + slop < value2)) {
    return(FALSE);
  } else {
    return(TRUE);
  }
}

/*************************************************************************
 * Convert a boolean to and from a "true" or "false" string.
 *************************************************************************/
char* boolean_to_string
 (BOOLEAN_T the_boolean)
{
  static char * true_or_false;
  static BOOLEAN_T first_time = TRUE;

  if (first_time) {
    true_or_false = (char *)mymalloc(sizeof(char) * 6);
    first_time = FALSE;
  }

  if (the_boolean) {
    strcpy(true_or_false, "true");
  } else {
    strcpy(true_or_false, "false");
  }
  return(true_or_false);
}

BOOLEAN_T boolean_from_string
  (char* true_or_false)
{
  if (strcmp(true_or_false, "true") == 0) {
    return(TRUE);
  } else if (strcmp(true_or_false, "false") == 0) {
    return(FALSE);
  } else {
    die("Invalid input to boolean_from_string (%s)\n", true_or_false);
  }
  return(FALSE); /* Unreachable. */
}


/**************************************************************************
 * Does a given character appear in a given string?
 **************************************************************************/
BOOLEAN_T char_in_string
  (const char* a_string,
   char        a_char)
{
  int  i_string;    /* Index into the string. */
  char string_char; /* Character appearing at that index. */

  i_string = 0;
  string_char = a_string[i_string];
  while (string_char != '\0') {
    if (string_char == a_char) {
      return(TRUE);
    }
    i_string++;
    string_char = a_string[i_string];
  }
  return(FALSE);
}

/**************************************************************************
 * Generic functions for converting between integer and string
 * representations of an enumerated type.
 *
 * Assumes that the longest string representation of the enumerated
 * type does not exceed 100 characters.
 *
 * Assumes that the zeroth enumerated type element is invalid.
 **************************************************************************/
char * convert_enum_type
  (int     enum_type, /* The enumerated type object to be converted. */
   char *  enum_strs[], /* String values associated with this type. */
   int     num_enums) /* Number of values of the type. */
{
  if ((enum_type <= 0) || (enum_type >= num_enums)) {
    die("Illegal enumerated type value (%d).", enum_type);
  }

  return(enum_strs[enum_type]);
}

int convert_enum_type_str
  (char *  enum_type_str, /* String to be converted. */
   int     default_value, /* Value to return if first arg is null. */
   char ** enum_strs,     /* String values associated with this type. */
   int     num_enums)     /* Number of values of the type. */
{
  int i_enum;

  /* If no string was given, return the default. */
  if (enum_type_str == NULL) {
    return(default_value);
  }

  /* Search for the value corresponding to the given string. */
  for (i_enum = 0; i_enum < num_enums; i_enum++) {
    if (strcmp(enum_type_str, enum_strs[i_enum]) == 0) {
      return(i_enum);
    }
  }
  die("Illegal value (%s).", enum_type_str);
  return(0); /* Unreachable. */
}


/****************************************************************************
 * Get the name of the CPU.
 ****************************************************************************/
#ifdef NOHOSTNAME
const char* hostname
  ()
{
  return("Unknown");
}
#else
#define HOST_LENGTH 100
const char* hostname
  ()
{
   static char the_hostname[HOST_LENGTH];

   if(gethostname(the_hostname, 255) != 0) {
     die("Could not get host information\n");
   }

   return(the_hostname);
}
#endif

/****************************************************************************
 * Get the current date and time.
 ****************************************************************************/
#include <time.h>
const char* date_and_time
  ()
{
  struct tm *tm_ptr;
  time_t current_time;
  static char buf[256];

  (void) time(&current_time);
  tm_ptr = localtime(&current_time);
  strftime(buf, 256, "%a %b %d %H:%M:%S %Z %Y", tm_ptr);

  return(buf);
}



/****************************************************************************
 * Copy a string, with allocation.
 ****************************************************************************/
char * copy_string
 (char ** target,
  char *  source)
{
  *target = (char *)mycalloc(strlen(source) + 1, sizeof(char));
  strcpy(*target, source);
  return(*target);
}

/*****************************************************************************
 * Shuffle an array of integers using Knuth's algorithm.
 *****************************************************************************/
static void shuffle
  (int   num_items,
   int*  data)
{
  int i_item;
  int i_rand;
  int temp_swap;

  // Shuffle the array.
  for (i_item = 0; i_item < num_items; i_item++) {

    // Select a random position to the right of the current position.
    i_rand = (int)(my_drand() * (double)(num_items - i_item)) + i_item;

    // Swap 'em.
    temp_swap = data[i_item];
    data[i_item] = data[i_rand];
    data[i_rand] = temp_swap;
  }
}

/*****************************************************************************
 * Randomly return an index between 0 and a given value.
 *
 * On the first call, current_item should be -1.
 *
 * The function allocates the data array and randomly fill it with
 * ascending integers.  On subsequent calls, iterate through the
 * array, returning the next index.  Once we iterate through the whole
 * thing, re-shuffle and start over.
 *****************************************************************************/
int select_random_item
  (int   num_items,
   int*  current_item,
   int** data)
{
  int i_item;
  int return_value;

  if (*current_item == -1) {
    *current_item = 0;

    /* Allocate memory for the array. */
    myfree(*data);
    *data = (int*)mymalloc(sizeof(int) * num_items);

    /* Fill the array with ascending integers. */
    for (i_item = 0; i_item < num_items; i_item++) {
      (*data)[i_item] = i_item;
    }
  }

  /* Shuffle if we're at the beginning of the array. */
  if (*current_item == 0) {
    shuffle(num_items, *data);
  }

  /* Store the return value. */
  return_value = (*data)[*current_item];

  /* Move to the next item. */
  (*current_item)++;
  if (*current_item >= num_items) {
    *current_item = 0;
  }

  /* Return. */
  return(return_value);
}

#ifdef UTILS_MAIN

int main (int argc, char *argv[])
{
  FILE *infile;
  char word[1000];
  long seed;
  int i, j;

  if (argc != 2) {
    die("USAGE: utils <filename>");
  }

  if (open_file(argv[1], "r", 1, "input", "", &infile) == 0)
    exit(1);

  while (fscanf(infile, "%s", word) == 1)
    printf("%s ", word);

  fclose(infile);

  /* Test the random number generator. */
  seed = time(0);
  my_srand(seed);
  printf("\nSome random numbers (seed=%ld): \n", seed);
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      printf("%6.4f ", my_drand());
    }
    printf("\n");
  }
  return(0);
}

#endif

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 2
 * End:
 */




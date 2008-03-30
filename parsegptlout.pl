#!/usr/bin/perl

use strict;

our ($verbose) = 0;   # output verbosity
our ($maxval);        # max value across all processes/threads
our ($minval);        # min value across all processes/threads
our ($sum);           # total
our ($nval);          # number of entries found across all processes/threads

my ($fn);             # file name
my ($fnroot) = "timing";
my ($target);         # region to search for
my ($arg);            # element of @ARGV
my ($col) = 1;        # column to use (default is wallclock)
my ($started);        # flag indicates initial "Stats for thread" found
my ($thread);         # thread number
my ($threadmax);      # thread number for max value
my ($threadmin);      # thread number for min value
my ($task);           # MPI task index (= ntask at loop completion)
my ($taskmax);        # task index for max value
my ($taskmin);        # task index for min value
my ($line);           # input line read in from file
my ($idx);            # index
my ($hidx);           # heading index
my ($mean);           # mean value
my ($found);          # flag indicates region name found

my (@vals);           # values for region
my (@heading);        # heading appropriate to col (e.g. "Wallclock")

# Parse arg list

while ($arg = shift (@ARGV)) {
    if ($arg eq "-v") {
	$verbose = 1;
    } elsif ($arg eq "-c") {
	$col = shift (@ARGV);       # set column
	($col > 0) || die_usemsg ("selected column ($col) must be 1 or greater\n");
    } elsif ($arg eq "-f") {
	$fnroot = shift (@ARGV);    # change root of file name
    } else {
	if ( ! defined $target ) {
	    $target = $arg;         # region name
	    chomp ($target);
	} else {
	    die_usemsg ("Unknown argument $arg\n");
	}
    }
}

die_usemsg ("Target region name not defined\n") if ( ! defined $target );
&initstats();     # Initialize stats
$found = 0;       # false
$idx = 1 + $col;  # index will always be the same
$hidx = 1 + $idx;     

# Loop through output files

for ($task = 0; -e "${fnroot}.$task"; $task++) {
    $fn = "${fnroot}.$task";
    open (FILE, "<$fn") || die ("Can't open $fn for reading\n");
    $started = 0;

# Read all the lines in the file, looking for "Stats for thread", followed by
# thre region name

    while ($line = <FILE>) {
	chomp ($line);
	if ($line =~ /^Stats for thread (\d*):/) {
	    $started = 1;
	    $thread = $1;

# Next line contains the headings. Parse for later printing

	    $line = <FILE>;
	    chomp ($line);
	    @heading = split (/\s+/, $line);
	} elsif ($started && ($line =~ /^[* ]\s*${target}\s*(.*)$/)) {
	    $found = 1;
	    @vals = split (/\s+/, $1);
	    print (STDOUT "vals=@vals\n") if ($verbose);
	    ($#vals >= $idx) || die ("No column $col found in input:\n$line\n");
	    $sum += $vals[$idx];
	    $nval++;
	    if ($vals[$idx] > $maxval) {
		$maxval = $vals[$idx];
		$taskmax = $task;
		$threadmax = $thread;
	    }
	    if ($vals[$idx] < $minval) {
		$minval = $vals[$idx];
		$taskmin = $task;
		$threadmin = $thread;
	    }
	    $started = 0;
	    next;   # Look for next "Stats for thread"
	}
    }
}

die ("Found no occurrences of $target in any of $task files\n") if ( ! $found );

print (STDOUT "Found $nval values spread across $task tasks\n");
print (STDOUT "Heading is $heading[$hidx]\n");
print (STDOUT "Max   =  $maxval on thread $threadmax task $taskmax\n");
print (STDOUT "Min   =  $minval on thread $threadmin task $taskmin\n");
$mean = $sum / $nval;
print (STDOUT "Mean  = $mean\n");
print (STDOUT "Total = $sum\n");

exit 0;

sub initstats {
    our ($verbose);
    our ($maxval);
    our ($minval);
    our ($sum);
    our ($nval);

    $minval = 9.99e19;
    $maxval = -9.99e19;
    $nval = 0;
    $sum = 0.;
    $taskmax = -1;
    $taskmin = -1;
    $threadmax = -1;
    $threadmin = -1;
}

sub die_usemsg {
    defined $_[0] && print (STDOUT "$_[0]");
    print (STDOUT "Usage: $0 [-v] [-c column] [-f file-root] region\n",
	   " -v           => verbose\n",
	   " -f file-root => look for files named <file-root>.<taskid>\n",
	   " -c column    => use numbers in this column\n",
	   " region       => region name to search for\n");
    exit 1;
}

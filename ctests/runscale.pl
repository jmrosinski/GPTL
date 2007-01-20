#!/usr/bin/perl

use Sys::Hostname;
use Cwd;

my ($host) = hostname;
my ($cmdfile) = "scale.gp";
my ($cmd);
@nodecounts = (2,3,4,6,8,9,10,11,12,13,14,16,18,22,26,30,32);
my ($nodecount);
my ($ret);
my (@fpopsfiles) = ("FPops_aggregate","FPops_max","FPops_min");
my (@membwfiles) = ("MemBW_aggregate","MemBW_max","MemBW_min");
my (@mpisendrecvfiles) = ("MPI_Sendrecv_aggregate","MPI_Sendrecv_max","MPI_Sendrecv_min");
my ($mpicmd) = "mpirun";
my ($rundir) = ".";
my ($arg);
my ($cwd);

# Remove old files

unlink (@fpopsfiles);
unlink (@membwfiles);
unlink (@mpisendrecvfiles);

# Parse arg list

chomp (@ARGV);
while (defined ($arg = shift (@ARGV))) {
    if ($arg eq "-help") {
	print (STDOUT "Usage: $0 [-help] [-mpicmd cmd] [-rundir dir]\n");
	exit 0;
    } elsif ($arg eq "-mpicmd") {
	defined ($mpicmd = shift (@ARGV)) || die ("Bad arg to -mpicmd\n");
    } elsif ($arg eq "-rundir") {
	defined ($rundir = shift (@ARGV)) || die ("Bad arg to -rundir\n");
	chdir $rundir                     || die ("Unable to chdir to $rundir\n");
    } else {
        die ("Unknown arg encountered: $arg\n");
    }
}

# Run the test cases

$cwd = getcwd;
print (STDOUT "Tests will be run in directory $cwd\n");

foreach $nodecount (@nodecounts) {
    $cmd = "$mpicmd -np $nodecount ./scale ";
    print (STDOUT "Running $cmd...\n");
    $ret = system ("$cmd");
    if ($ret != 0) {
	print (STDOUT "Bad return from $cmd: $ret: exiting\n");
	exit ($ret);
    }
}

# Build the base gnuplot command file

open (CMDFILE, ">$cmdfile") || die ("Can't open $cmdfile for writing\n");

print (CMDFILE "set title \"$host\"\n");
print (CMDFILE "set key left\n");
print (CMDFILE "set style data linespoints\n");
print (CMDFILE "set terminal postscript color\n");
print (CMDFILE "set xrange [0:*]\n");
print (CMDFILE "set xtics 10\n");
print (CMDFILE "set mxtics 10\n");
print (CMDFILE "set xlabel \"Number of MPI tasks\"\n");

# Define linear speedup from first point

&getlinear ($fpopsfiles[0], "fpopslinear");
print (CMDFILE "set ylabel \"MFlops/sec\"\n");
print (CMDFILE "set output 'FPops.ps'\n");
print (CMDFILE "plot '$fpopsfiles[0]' using 1:2, '$fpopsfiles[1]' using 1:2, '$fpopsfiles[2]' using 1:2, 'fpopslinear' using 1:2 with lines\n");

&getlinear ($membwfiles[0], "membwlinear");
print (CMDFILE "set ylabel \"MB/sec\"\n");
print (CMDFILE "set output 'MemBW.ps'\n");
print (CMDFILE "plot '$membwfiles[0]' using 1:2, '$membwfiles[1]' using 1:2, '$membwfiles[2]' using 1:2, 'membwlinear' using 1:2 with lines\n");

&getlinear ($mpisendrecvfiles[0], "mpisendrecvlinear");
print (CMDFILE "set ylabel \"MB/sec\"\n");
print (CMDFILE "set output 'MPI_Sendrecv.ps'\n");
print (CMDFILE "plot '$mpisendrecvfiles[0]' using 1:2, '$mpisendrecvfiles[1]' using 1:2, '$mpisendrecvfiles[2]' using 1:2, 'mpisendrecvlinear' using 1:2 with lines\n");

close (CMDFILE);

exit (0);

sub getlinear
{
    my ($fn) = $_[0];
    my ($linearfn) = $_[1];
    my ($max);

    my ($refline);
    my ($dum);
    my ($ref);

    open (REFCOUNT, "<$fn") || die ("Can't open $fn for reading\n");
    $refline = <REFCOUNT>;
    chomp ($refline);
    close (REFCOUNT);
    ($dum, $ref) = split (/\s+/, $refline);
    $max = ($nodecounts[$#nodecounts] / $nodecounts[0]) * $ref;
    open (LINEAR, ">$linearfn") || die ("Can't open $linearfn for writing\n");
    print (LINEAR "$nodecounts[0] $ref\n");
    print (LINEAR "$nodecounts[$#nodecounts] $max\n");
    close (LINEAR);
}

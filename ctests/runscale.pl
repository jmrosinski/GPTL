#!/usr/bin/perl

use Sys::Hostname;
use Cwd;

my ($host) = hostname;
my ($cmdfile) = "scale.gp";
my ($cmd);
@taskcounts = (2,4,8,16,24);
#@taskcounts = (1,2,3);
my ($taskcountrange) = $taskcounts[$#taskcounts] - $taskcounts[0];
my ($taskcount);
my ($N);
my ($p);
my ($ret);

my (@fpopsfiles) = ("FPOPS_max","FPOPS_min");
my (@membwfiles) = ("MEMBW_max","MEMBW_min");

my (@sendrecvbasefiles) = ("Sendrecv_base_max","Sendrecv_base_min");
my (@sendrecvfabricfiles) = ("Sendrecv_fabric_max","Sendrecv_fabric_min");
my (@sendrecvmemoryfiles) = ("Sendrecv_memory_max","Sendrecv_memory_min");

my (@isendirecvbasefiles) = ("IsendIrecv_base_max","IsendIrecv_base_min");
my (@isendirecvfabricfiles) = ("IsendIrecv_fabric_max","IsendIrecv_fabric_min");
my (@isendirecvmemoryfiles) = ("IsendIrecv_memory_max","IsendIrecv_memory_min");

my (@irecvisendbasefiles) = ("IrecvIsend_base_max","IrecvIsend_base_min");
my (@irecvisendfabricfiles) = ("IrecvIsend_fabric_max","IrecvIsend_fabric_min");
my (@irecvisendmemoryfiles) = ("IrecvIsend_memory_max","IrecvIsend_memory_min");

my ($mpicmd) = "srun";
#my ($mpicmd) = "mpiexec";
my ($rundir) = ".";
my ($arg);
my ($cwd);
my ($xtics);

# Remove old files

unlink (@fpopsfiles);
unlink (@membwfiles);

unlink (@sendrecvbasefiles);
unlink (@sendrecvfabricfiles);
unlink (@sendrecvmemoryfiles);

unlink (@isendirecvbasefiles);
unlink (@isendirecvfabricfiles);
unlink (@isendirecvmemoryfiles);

unlink (@irecvisendbasefiles);
unlink (@irecvisendfabricfiles);
unlink (@irecvisendmemoryfiles);

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

foreach $taskcount (@taskcounts) {
    $N = $taskcount;
    $N = 4 if ($N > 4);
    $p = $taskcount / $N;
    $cmd = "$mpicmd -n $taskcount -N $N ./scale -p $p  -l 1000000 -n 10";
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
print (CMDFILE "set yrange [0:*]\n");
if ($taskcountrange < 10) {
    $xtics = 1;
} elsif ($taskcountrange < 100) {
    $xtics = 10;
} elsif ($taskcountrange < 1000) {
    $xtics = 100;
} else {
    $xtics = 1000;
}
print (CMDFILE "set xtics $xtics\n");
if ($xtics > 1) {
    print (CMDFILE "set mxtics 10\n");
}
print (CMDFILE "set xlabel \"Number of MPI tasks\"\n");

# Define linear speedup from first point

if (&getlinear ($fpopsfiles[0], "FPOPS_linear")) {
    print (CMDFILE "set ylabel \"MFlops/sec\"\n");
    print (CMDFILE "set output 'FPOPS.ps'\n");
    print (CMDFILE "plot \'$fpopsfiles[0]\' using 1:2, \'$fpopsfiles[1]\' using 1:2, 'FPOPS_linear' using 1:2 with lines\n");
}

if (&getlinear ($membwfiles[0], "MEMBW_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'MEMBW.ps'\n");
    print (CMDFILE "plot \'$membwfiles[0]\' using 1:2, \'$membwfiles[1]\' using 1:2, 'MEMBW_linear' using 1:2 with lines\n");
}


if (&getlinear ($sendrecvbasefiles[0], "Sendrecv_base_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'Sendrecv_base.ps'\n");
    print (CMDFILE "plot \'$sendrecvbasefiles[0]\' using 1:2, \'$sendrecvbasefiles[1]\' using 1:2, \'Sendrecv_base_linear\' using 1:2 with lines\n");
}

if (&getlinear ($sendrecvfabricfiles[0], "Sendrecv_fabric_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'Sendrecv_fabric.ps'\n");
    print (CMDFILE "plot \'$sendrecvfabricfiles[0]\' using 1:2, \'$sendrecvfabricfiles[1]\' using 1:2, 'Sendrecv_fabric_linear' using 1:2 with lines\n");
} 

if (&getlinear ($sendrecvmemoryfiles[0], "Sendrecv_memory_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'Sendrecv_memory.ps'\n");
    print (CMDFILE "plot \'$sendrecvmemoryfiles[0]\' using 1:2, \'$sendrecvmemoryfiles[1]\' using 1:2, 'Sendrecv_memory_linear' using 1:2 with lines\n");
}


if (&getlinear ($isendirecvbasefiles[0], "IsendIrecv_base_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IsendIrecv_base.ps'\n");
    print (CMDFILE "plot \'$isendirecvbasefiles[0]\' using 1:2, \'$isendirecvbasefiles[1]\' using 1:2, 'IsendIrecv_base_linear' using 1:2 with lines\n");
}

if (&getlinear ($isendirecvfabricfiles[0], "IsendIrecv_fabric_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IsendIrecv_fabric.ps'\n");
    print (CMDFILE "plot \'$isendirecvfabricfiles[0]\' using 1:2, \'$isendirecvfabricfiles[1]\' using 1:2, 'IsendIrecv_fabric_linear' using 1:2 with lines\n");
}

if (&getlinear ($isendirecvmemoryfiles[0], "IsendIrecv_memory_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IsendIrecv_memory.ps'\n");
    print (CMDFILE "plot \'$isendirecvmemoryfiles[0]\' using 1:2, \'$isendirecvmemoryfiles[1]\' using 1:2, 'IsendIrecv_memory_linear' using 1:2 with lines\n");
}


if (&getlinear ($irecvisendbasefiles[0], "IrecvIsend_base_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IrecvIsend_base.ps'\n");
    print (CMDFILE "plot \'$irecvisendbasefiles[0]\' using 1:2, \'$irecvisendbasefiles[1]\' using 1:2, 'IrecvIsend_base_linear' using 1:2 with lines\n");
}
 
if (&getlinear ($irecvisendfabricfiles[0], "IrecvIsend_fabric_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IrecvIsend_fabric.ps'\n");
    print (CMDFILE "plot \'$irecvisendfabricfiles[0]\' using 1:2, \'$irecvisendfabricfiles[1]\' using 1:2, 'IrecvIsend_fabric_linear' using 1:2 with lines\n");
}

if (&getlinear ($irecvisendmemoryfiles[0], "IrecvIsend_memory_linear")) {
    print (CMDFILE "set ylabel \"MB/sec\"\n");
    print (CMDFILE "set output 'IrecvIsend_memory.ps'\n");
    print (CMDFILE "plot \'$irecvisendmemoryfiles[0]\' using 1:2, \'$irecvisendmemoryfiles[1]\' using 1:2, 'IrecvIsend_memory_linear' using 1:2 with lines\n");
}

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

    if (open (REFCOUNT, "<$fn")) {
	$refline = <REFCOUNT>;
	chomp ($refline);
	close (REFCOUNT);
	($dum, $ref) = split (/\s+/, $refline);
	$max = ($taskcounts[$#taskcounts] / $taskcounts[0]) * $ref;
	open (LINEAR, ">$linearfn") || die ("Can't open $linearfn for writing\n");
	print (LINEAR "$taskcounts[0] $ref\n");
	print (LINEAR "$taskcounts[$#taskcounts] $max\n");
	close (LINEAR);
	return 1;
    } else {
	print (STDOUT "Can't open $fn for reading\n");
	return 0;
    }
}

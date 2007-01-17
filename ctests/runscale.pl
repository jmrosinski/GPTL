#!/usr/bin/perl

use strict;
use Sys::Hostname;

my ($host) = hostname;
my ($cmdfile) = "scale.gp";
my ($cmd);
my (@nodecounts) = (1,2,3,4);
my ($nodecount);
my ($ret);
my ($refline);
my ($dum);
my ($ref);
my ($max);
my (@fpopsfiles) = ("FPops_aggregate","FPops_max","FPops_min");

unlink (@fpopsfiles);

foreach $nodecount (@nodecounts) {
    $cmd = "mpiexec -n $nodecount ./scale";
    print (STDOUT "Running $cmd...\n");
    $ret = system ("$cmd");
}

# Define linear speedup from first point

open (REFCOUNT, "<$fpopsfiles[0]") || die ("Can't open $fpopsfiles[0] for reading\n");
$refline = <REFCOUNT>;
chomp ($refline);
close (REFCOUNT);
($dum, $ref) = split (/\s+/, $refline);
$max = ($nodecounts[$#nodecounts] / $nodecounts[0]) * $ref;
open (LINEAR, ">linear") || die ("Can't open linear for writing\n");
print (LINEAR "$nodecounts[0] $ref\n");
print (LINEAR "$nodecounts[$#nodecounts] $max\n");

open (CMDFILE, ">$cmdfile") || die ("Can't open $cmdfile for writing\n");
print (CMDFILE "set title \"$host\"\n");
print (CMDFILE "set style data linespoints\n");
print (CMDFILE "set terminal postscript color\n");
print (CMDFILE "set output 'FPops.ps'\n");
print (CMDFILE "set xrange [1:*]\n");
print (CMDFILE "set xlabel \"Number of MPI tasks\"\n");
print (CMDFILE "set ylabel \"MFlops/sec\"\n");
print (CMDFILE "plot '$fpopsfiles[0]' using 1:2, '$fpopsfiles[1]' using 1:2, '$fpopsfiles[2]' using 1:2, 'linear' using 1:2 with lines\n");
close (CMDFILE);

exit (0);

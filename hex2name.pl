#!/usr/bin/perl

# jr-resolve.pl - convert timing lib output addresses to names
# hacked from cyg-profile script found on web.

use strict;
use warnings;
no warnings 'portable';
use diagnostics;
use English;

my (%symtab);    # symbol table derived from executable
my ($binfile);   # executable
my ($timingout); # timer file (normally timing.[0-9]*)
my ($demangle);  # whether to demangle the symbols
my ($arg);       # cmd-line arg
my ($PRTHRESH) = 1000000; # This needs to match what is in the GPTL lib

$OUTPUT_AUTOFLUSH = 1;

&help() if ($#ARGV < 1 || $#ARGV > 2);

while ($arg = shift (@ARGV)) {
    if ($arg eq "-demangle") {
	$demangle = 1;
    } elsif ( ! defined ($binfile)) {
	$binfile = $arg;
    } else {
	$timingout = $arg;
    }
}

&help() if ($binfile =~ /--help/);
&help() if (!defined ($binfile));
&help() if (!defined ($timingout));

&main();

# ==== Subs

sub help()
{
    printf ("Usage: $0 [-demangle] executable timing_file\n");
    exit;
}

sub main()
{
    my ($offset);    # offset into a.out (to match timing output)
    my ($type);      # symbol type
    my ($function);  # name of function in symtab
    my ($nsym) = 0;  # number of symbols
    my ($nfunc) = 0; # number of functions
    my ($sym);       # symbol
    my ($begofline);
    my ($off1);
    my ($spaftsym);
    my ($ncalls);
    my ($restofline);
    my ($numsp);  # number of spaces before rest of line
    my ($spaces);     # text containing spaces before rest of line
    my ($thread) = -1;   # thread number (init to -1
    my ($doparse) = 0;   # logical flag: true indicates between "Statas for thread..."
                         # and "Number of calls..."
    my ($indent);
    my (@max_chars);     # longest symbol name + indentation (per thread)
	
    if ($demangle) {
	open (NM, "nm $binfile | c++filt | ") or die ("Unable to run 'nm $binfile | c++filt': $!\n");
    } else {
	open (NM, "nm $binfile |") or die ("Unable to run 'nm $binfile': $!\n");
    }

    printf ("Loading symbols from $binfile ... ");
	
    while (<NM>) {
	$nsym++;
	next if (!/^([0-9A-F]+) (.) (.+)$/i);
	$offset   = hex($1); 
	$type     = $2; 
	$function = $3;
	next if ($type !~ /[tT]/);
	$nfunc++;
	$symtab{$offset} = $function;
    }
    printf("OK\nSeen %d symbols, stored %d function offsets\n", $nsym, $nfunc);
    close(NM);

    @max_chars = &get_max_chars ("$timingout");

    open (TEXT, "<$timingout") or die ("Unable to open '$timingout': $!\n");
	
    while (<TEXT>) {

	# Parse the line if it's a hex number followed by a number
	
	if (/Stats for thread /) { # beginning of main region
	    $doparse = 1;
	    ++$thread;
	    print $_; 
	} elsif (/^Total calls /) {  # end of main region
	    $doparse = 0;
	    print $_; 
	} elsif ($doparse) {                # Inside main region
	    if (/^ *(Called  Recurse.*)$/) { # heading
		$numsp = $max_chars[$thread];
		$spaces = " " x $numsp;
		printf ("%s   %s\n", $spaces, $1);
	    } elsif (/(^\*? *)([[:xdigit:]]+)( +)([0-9.Ee+]+)(.*)$/) { # hex entry
		$begofline  = $1;
		$off1       = hex($2);
		$ncalls     = $4;
		$restofline = $5;
		if (defined ($symtab{$off1})) {
		    $sym = $symtab{$off1};
		} else {
		    $sym = "???";
		}
		$numsp = $max_chars[$thread] - length ($begofline) - length ($sym);
		$spaces = " " x $numsp;
		printf ("%s%s%s %9s %s\n", $begofline, $sym, $spaces, $ncalls, $restofline);
	    } elsif (/(^\*? *)(\w+)( +)([0-9.Ee+]+)(.*)$/) { # standard entry
		$begofline  = $1;
		$sym        = $2;
		$ncalls     = $4;
		$restofline = $5;
		$numsp = $max_chars[$thread] - length ($begofline) - length ($sym);
		$spaces = " " x $numsp;
		printf ("%s%s%s %9s %s\n", $begofline, $sym, $spaces, $ncalls, $restofline);
	    } else {           # unknown: just print it
		print $_; 
	    }
	} elsif (/(^ *)([0-9.Ee+]+)( +)([[:xdigit:]]+)( *)$/) {
#
# Hex entry in multiple parent region
#
	    $ncalls     = $2;
	    $indent     = $3;
	    $off1       = hex($4);
	    if (defined ($symtab{$off1})) {
		$sym = $symtab{$off1};
	    } else {
		$sym = "???";
	    }
	    $restofline = $5;
	    printf ("%8s%s%s%s\n", $ncalls, $indent, $sym, $restofline);
	} else { # unknown: just print it
	    print $_; 
	    next;
	}
    }
    close (TEXT);
    printf("done\n");
}

sub get_max_chars ()
{
    my ($file) = $_[0];
    my ($thread) = -1;
    my ($tmp);
    my ($sym);
    my ($off1);
    my ($doparse) = 0;
    my (@max_chars);
    
    open (TEXT, "<$file") or die ("Unable to open '$file': $!\n");
    
    while (<TEXT>) {

	# Parse the line if it's a hex number followed by a number
	# Otherwise just write it
	
	if (/Stats for thread /) {
	    $doparse = 1;
	    ++$thread;
	    $max_chars[$thread] = 0;
	} elsif (/^Total calls /) {
	    $doparse = 0;
	} elsif ($doparse && /(^\*? *)([[:xdigit:]]+)/) {
	    $off1 = hex($2);
	    if (defined ($symtab{$off1})) {
		$sym = $symtab{$off1};
	    } else {
		$sym = "???";
	    }
	    $tmp = length ($1) + length ($sym);
	    if ($tmp > $max_chars[$thread]) {
		$max_chars[$thread] = $tmp;
	    }
	}
    }
    close (TEXT);
    return @max_chars;
}

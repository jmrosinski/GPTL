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
    my ($newlen3);
    my ($new3);
	
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

    open (TEXT, "$timingout") or die ("Unable to open '$timingout': $!\n");
	
    while (<TEXT>) {

	# Parse the line if it's a hex number followed by a number
	# Otherwise just write it

	if (/(^ *)([[:xdigit:]]+)( *)([[:digit:]]+)(.*)/) {
	    $begofline  = $1;
	    $off1       = hex($2);
	    $spaftsym   = $3;
	    $ncalls     = $4;
	    $restofline = $5;
	    if (defined ($symtab{$off1})) {
		$sym = $symtab{$off1};
	    } else {
		$sym = "???";
	    }
#	    $sym = ( ? $symtab{$off1} : "???");

# Attempt to line things up--won't work if length of mangled name
# is too long.

	    $newlen3 = length ($3) + (length ($2) - length ($sym));
	    if ($newlen3 > 0) {
		$new3 = " " x $newlen3;
		printf ("%s%s%s%s%s\n", $begofline, $sym, $new3, $ncalls, $restofline);
	    } else {
		printf ("%s%s %s%s\n", $begofline, $sym, $ncalls, $restofline);
	    }
	} else {
	    print $_; 
	    next;
	}
    }
    close (TEXT);
    printf("done\n");
}

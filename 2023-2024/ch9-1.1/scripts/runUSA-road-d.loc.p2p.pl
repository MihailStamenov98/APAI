# ============================================================================
#  runUSA-road-d.loc.p2p.pl
# ============================================================================

#  Author(s)       (c) 2006 Camil Demetrescu, Andrew Goldberg
#  License:        See the end of this file for license information
#  Created:        Feb 15, 2006

#  Last changed:   $Date: 2006/10/30 08:09:39 $
#  Changed by:     $Author: demetres $
#  Revision:       $Revision: 1.2 $

#  9th DIMACS Implementation Challenge: Shortest Paths
#  http://www.dis.uniroma1.it/~challenge9

#  USA-road-d family experiment driver
#  runs the p2p solver on instances in the USA-road-d family

#  Usage: > perl runUSA-road-d.loc.p2p.pl
# ============================================================================

# param setup:
$RESFILE   = "../results/USA-road-d.loc.p2p.res";
$PREFIX    = "../inputs/USA-road-d/USA-road-d";
$SOLVER    = "../solvers/mlb-dimacs/mbp.exe";

$GRAPH     = "$PREFIX.%s.gr";
$AUX       = "$PREFIX.%s.%s.p2p";

# header:
print "\n* 9th DIMACS Implementation Challenge: Shortest Paths\n";
print   "* http://www.dis.uniroma1.it/~challenge9\n";
print   "* USA-road-d family loc p2p core experiment\n";

# open result file
open FILE, ">$RESFILE" or die "Cannot open $RESFILE for write :$!";

# generation subroutine
sub DORUN {

    # graph instance (e.g., CTR, BAY, etc.)
    $EXT = $_[0];

    # run experiments for different degrees of locality of queries
    for ( $baserank = 0; $baserank < 64; ++$baserank) {

        $graphname = sprintf $GRAPH, $EXT;
        $auxname   = sprintf $AUX, $EXT, $baserank;

        # if aux file with given rank exists, run experiment and collect stdout
        if (-e $auxname) {
            print "\n* Running p2p solver on graph $graphname\n* Problem instance: $auxname\n";
            $out = `$SOLVER $graphname $auxname`;
            chop($out);
            $out = sprintf "$out %4s\n", $baserank;
            print FILE $out;				
        }
    }
}

#create instances
#&DORUN("NY");
#&DORUN("BAY");
#&DORUN("COL");
#&DORUN("FLA");
#&DORUN("NW");
#&DORUN("NE");
#&DORUN("CAL");
&DORUN("LKS");
#&DORUN("E");
#&DORUN("W");
#&DORUN("CTR");

close FILE;


# ============================================================================
# Copyright (C) 2006 Camil Demetrescu, Andrew Goldberg

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
# ============================================================================


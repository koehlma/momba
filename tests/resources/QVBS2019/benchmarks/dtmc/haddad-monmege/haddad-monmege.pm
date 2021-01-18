// Toy example on which standard value iteration yields wrong results
// This model originates from Haddad, Monmege: Reachability in MDPs: Refining Convergence of Value Iteration

dtmc

const int N;
const double p;
const double q = 0.5;

module main
	x : [0..2*N] init N;
	[] x=N -> p : (x'=N-1) + (1-p) : (x'=N+1);
	[] x>0 & x<N -> q : (x'=x-1) + (1-q) : (x'=N);
	[] x>N & x<2*N -> q : (x'=x+1) + (1-q) : (x'=N);
	[] x=0 | x=2*N -> 1 : true;
endmodule

label "Target" = x=0;
label "Done" = x=0 | x=2*N;

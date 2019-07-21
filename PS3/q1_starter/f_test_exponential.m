function val = f_test(x)

A =[ ...
   -0.4348    0.4694    0.5529    1.0184   -1.2344
   -0.0793   -0.9036   -0.2037   -1.5804    0.2888
    1.5352    0.0359   -2.0543   -0.0787   -0.4293
   -0.6065   -0.6275    0.1326   -0.6817    0.0558
   -1.3474    0.5354    1.5929   -1.0246   -0.3679
];

b = [ ...
	-3.0319
    1.8434
    1.5232
    5.0773
    1.7738];


if(length(x) ~= 5)
	'error'
end

a = A*x + b;

val = sum(exp(a));








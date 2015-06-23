this simple benchmark shows a design flaw into Boost / C++11 containers

Basically it shows an empty "std::unordered_map" or "boost::unordered_map"
can use an arbitrary amount of memory even when there are no elements stored.

Output:
allocated memory before returning 1056323

Don't forget to Starr this repository to gain attention and obtain a quicker fix.
Also take a look at status here:

https://github.com/Darelbi/Boost.EnanchedMap/blob/master/README.md

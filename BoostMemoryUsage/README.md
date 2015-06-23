this simple benchmark shows a design flaw into Boost / C++11 containers


Basically it shows an empty "std::unordered_map" or "boost::unordered_map"
can use an arbitrary amount of memory even when there are no elements stored.

Such behaviour is not documented and must be "profiled" to see that.

We miss a method for releasing memory => that would be the most 
maintainable solution because people using unordered_map have to
either wrap the map in something else adding more code,
or alternatively make it going out of scope.

Using such method would be convenient, alternatively documentation
must state it clearly that bad usage of containers may
use MUCH more memory than expected.


Output:
allocated memory before returning 1056323
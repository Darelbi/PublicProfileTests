To view code click on the file "UnorderedMap.cpp" above 

# design flaw into Boost / C++11 's unordered_map

An empty "std::unordered_map" or "boost::unordered_map"
can use an arbitrary amount of memory even when there are no elements stored.


Output:
```
allocated memory before returning 1056323
```

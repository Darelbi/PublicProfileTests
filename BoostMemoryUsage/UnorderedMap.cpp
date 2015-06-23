#include <unordered_map> //c++ container but in practice the same of boost
#include <memory>
#include <iostream>
using namespace std;

size_t counter = 0;

template <typename T>
class countingAllocator: public std::allocator<T>
{
public:
	typedef size_t size_type;
	typedef T* pointer;
	typedef const T* const_pointer;

	template<typename _Tp1>
	struct rebind
	{
			typedef countingAllocator<_Tp1> other;
	};

	pointer allocate(size_type n, const void *hint=0){
		   counter += n;
			return std::allocator<T>::allocate(n, hint);
	}

	void deallocate(pointer p, size_type n){
			counter -= n;
			return std::allocator<T>::deallocate(p, n);
	}
	
	static size_t getAllocatedBytes() { return counter;}

	countingAllocator() throw(): std::allocator<T>() {}
	countingAllocator(const countingAllocator &a) throw(): std::allocator<T>(a) { }
	template <class U>                    
	countingAllocator(const countingAllocator<U> &a) throw(): std::allocator<T>(a) { }
	~countingAllocator() throw() { }
};

template <int N>
struct big_struct {
	char c[N];
};

template<

	class Key,
	class T,
	class Hash = std::hash<Key>,
	class KeyEqual = std::equal_to<Key>,
	class Allocator = std::allocator< std::pair<const Key, T> >
> class unordered_map;

int main( int argc, char ** argv) {
	typedef big_struct<64> data_type;
	typedef std::unordered_map<int, data_type*, std::hash<int>, std::equal_to<int>, 
							countingAllocator< std::pair< const int, data_type*>> > map_type;

	map_type m;

	for (int i = 0; i < 1000 * 1000; i++) {
			m.insert(std::make_pair(i, new data_type));
	}   

	for (map_type::iterator it = m.begin(); it != m.end();) {
			delete it->second;
			it = m.erase(it);
	}   

	std::cout << "allocated memory before returning " << countingAllocator< std::pair< const int, data_type*>> ::getAllocatedBytes() << std::endl;

	return 0;
}
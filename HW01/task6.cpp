#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cout << "Invalid parameters" << endl;
		return 1;
	}
	int N = atoi(argv[1]);
	for (int i = 0; i < N+1; i++) {
		printf("%d ", i);
	}
	cout << "\n";
	for (int i = N; i >= 0; i--) {
		cout << i << " ";
	}
	cout << "\n";
	return 0;
}

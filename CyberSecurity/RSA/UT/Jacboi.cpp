#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;

// Function to calculate the Jacobi symbol (r/p)
int jacobi(int r, int p) 
{

    // Calculate the equations 
    double Odd_power= (r - 1) * (p - 1) / 4; 
    double Even_power = (p * p - 1) / 8;
    if (r == 0) return (p == 1) ? 1 : 0;  // Base case: Jacobi symbol is 0 if r is 0, unless p is 1
    if (r == 1) return 1;  // Base case: Jacobi symbol is 1 if r is 1
    int result = 1;
    if (r % 2 == 0) 
    {  // Check if r is even
        result = jacobi(r / 2, p);  // Recursively calculate Jacobi symbol for half of r
        result = result*pow(-1,Even_power);  // Adjust sign 
    } else 
    {  // When r is odd
        result = jacobi(p % r, r);  // Recursively calculate Jacobi symbol for p mod r
        result = result*pow(-1,Odd_power);   // Adjust sign 
    }
    return result;
}

// Function to perform modular exponentiation
int mod_exp(int base, int exp, int mod) 
{
    int result = 1;
    base = base % mod;  // Ensure base is within modulo
    while (exp > 0) 
    {  // Loop until exponent becomes 0
        if (exp % 2 == 1)  // If exponent is odd
            result = (result * base) % mod;  // Update result
        exp = exp >> 1;  // Divide exponent by 2
        base = (base * base) % mod;  // Square the base and reduce modulo
    }
    return result;
}

// Function to compute GCD (Greatest Common Divisor) of a and b
int gcd(int a, int b) 
{
    while (b != 0) 
    {  // Loop until b becomes 0
        int t = b;  // Store b in temporary variable
        b = a % b;  // Update b to a modulo b
        a = t;  // Update a to temporary variable
    }
    return a;  // Return the GCD
}

// Function to perform the Solovay-Strassen primality test
bool is_prime(int p, int k = 5) 
{
    if (p == 2) return true;  // 2 is prime
    if (p < 2 || p % 2 == 0) return false;  // No even number less than 2 is prime

    for (int i = 0; i < k; ++i) 
    {
        int r = rand() % (p - 2) + 2;  // Random integer in range [2, p-1]
        
        // Check if r and p are coprime
        if (gcd(r, p) != 1)  return false;
        
        int jacobian = (p + jacobi(r, p)) % p;  // Calculate Jacobi symbol and adjust to be positive
        int mod = mod_exp(r, (p - 1) / 2, p);  // Compute r^((p-1)/2) mod p

        if (jacobian == 0 || mod != jacobian)  // If condition fails, p is composite
            return false;
    }
    return true;  // Probably prime if all tests passed
}

// Main function to test the primality of a number
int main() 
{
    srand(time(0));  // Seed for random number generator

    int num = 13;  // Example number to test
    cout << "Is " << num << " a prime number? " << (is_prime(num) ? "Yes" : "No") << endl;
    
    return 0;
}

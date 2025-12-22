#include <iostream>
using namespace std;

/**
 * Function to compute the GCD and coefficients (x, y) such that:
 * ax + by = gcd(a, b)
 * @param FirstInteger              - First integer
 * @param SecondInteger             - Second integer
 * @param Coefficient_FirstInteger  - Reference to coefficient First integer
 * @param Coefficient_SecondInteger - Reference to coefficient SecondInteger
 * @return gcd(FirstInteger, SecondInteger)
 */
int extendedEuclidean(int FirstInteger, int SecondInteger, int &Coefficient_FirstInteger, int &Coefficient_SecondInteger) {
    // Base case: if b is 0, gcd is a, and x = 1, y = 0
    if (SecondInteger == 0) {
        Coefficient_FirstInteger  = 1;
        Coefficient_SecondInteger = 0;
        return FirstInteger;
    }

    // Recursive call
    int x1, y1; // Temporary coefficients
    int gcd = extendedEuclidean(SecondInteger, FirstInteger % SecondInteger, x1, y1);

    // Update x and y using the results of the recursive call
    Coefficient_FirstInteger = y1;
    Coefficient_SecondInteger = x1 - (FirstInteger / SecondInteger) * y1;

    return gcd;
}

/**
 * Function to compute the modular inverse of a modulo m using the Extended Euclidean Algorithm.
 * The modular inverse exists if and only if gcd(a, m) = 1.
 * @param PublicExponent - The number to find the inverse of
 * @param Phi - The modulus
 * @return The modular inverse of a modulo m, or -1 if no inverse exists
 */
int modularInverse(int PublicExponent, int Phi) 
{
    int x, y;
    int gcd = extendedEuclidean(PublicExponent, Phi, x, y);

    // If gcd(a, m) != 1, modular inverse does not exist
    if (gcd != 1) {
        cout << "Modular inverse does not exist for " << PublicExponent << " modulo " << Phi << endl;
        return -1;
    }

    // Ensure the result is positive
    return (x % Phi + Phi) % Phi;
}

int main() {
    // Example usage for RSA

    // Example inputs: public key exponent 'e' and modulus 'phi'
    int e = 7;     // Public exponent (coprime with phi)
    int phi = 40;  // Euler's totient function value for RSA

    // Find the modular inverse of e modulo phi
    int d = modularInverse(e, phi);

    if (d != -1) {
        cout << "The modular inverse of " << e << " modulo " << phi << " is: " << d << endl;
    }

    return 0;
}

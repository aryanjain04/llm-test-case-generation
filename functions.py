import re

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def divide(a, b):
    return a / b

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def factorial(n):
    if n < 0:
        raise ValueError("Negative input")
    if n == 0:
        return 1
    return n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None

def max_of_three(a, b, c):
    return max(a, b, c)

def sort_list(lst):
    return sorted(lst)

def password_strength(password):
    if len(password) < 8:
        return "weak"
    if not re.search(r"[A-Z]", password):
        return "weak"
    if not re.search(r"[0-9]", password):
        return "weak"
    return "strong"

import pytest
import re
from functions import *

# --- Tests for is_prime ---
class TestIsPrime:
    @pytest.mark.parametrize("n, expected", [
        (2, True),
        (13, True),
        (4, False),
        (1, False),
        (0, False),
        (-5, False),
        (97, True),
    ])
    def test_is_prime_variants(self, n, expected):
        assert is_prime(n) == expected

# --- Tests for divide ---
class TestDivide:
    def test_divide_normal(self):
        assert divide(10, 2) == 5.0

    def test_divide_float(self):
        assert divide(5, 2) == 2.5

    def test_divide_negative(self):
        assert divide(-10, 2) == -5.0

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)

# --- Tests for gcd ---
class TestGcd:
    @pytest.mark.parametrize("a, b, expected", [
        (48, 18, 6),
        (101, 10, 1),
        (0, 5, 5),
        (0, 0, 0),
        (-12, 8, 4),
    ])
    def test_gcd_variants(self, a, b, expected):
        assert gcd(a, b) == expected

# --- Tests for factorial ---
class TestFactorial:
    def test_factorial_zero(self):
        assert factorial(0) == 1

    def test_factorial_normal(self):
        assert factorial(5) == 120

    def test_factorial_negative(self):
        with pytest.raises(ValueError, match="Negative input"):
            factorial(-1)

# --- Tests for is_palindrome ---
class TestIsPalindrome:
    @pytest.mark.parametrize("s, expected", [
        ("racecar", True),
        ("A man a plan a canal Panama", True),
        ("hello", False),
        ("", True),
        ("121", True),
    ])
    def test_palindrome_variants(self, s, expected):
        assert is_palindrome(s) == expected

# --- Tests for binary_search ---
class TestBinarySearch:
    def test_binary_search_found(self):
        assert binary_search([1, 2, 3, 4, 5], 3) == 2

    def test_binary_search_not_found(self):
        assert binary_search([1, 2, 3, 4, 5], 6) == -1

    def test_binary_search_empty(self):
        assert binary_search([], 1) == -1

    def test_binary_search_first_last(self):
        arr = [10, 20, 30]
        assert binary_search(arr, 10) == 0
        assert binary_search(arr, 30) == 2

# --- Tests for validate_email ---
class TestValidateEmail:
    @pytest.mark.parametrize("email, expected", [
        ("test@example.com", True),
        ("user.name@domain.co.uk", True),
        ("invalid-email", False),
        ("test@domain", False),
        ("@missing.com", False),
        ("test@.com", False),
    ])
    def test_email_variants(self, email, expected):
        assert validate_email(email) == expected

# --- Tests for max_of_three ---
class TestMaxOfThree:
    def test_max_of_three_variants(self):
        assert max_of_three(1, 2, 3) == 3
        assert max_of_three(10, 5, 2) == 10
        assert max_of_three(-1, -5, -3) == -1
        assert max_of_three(5, 5, 5) == 5

# --- Tests for sort_list ---
class TestSortList:
    def test_sort_list_normal(self):
        assert sort_list([3, 1, 4, 2]) == [1, 2, 3, 4]

    def test_sort_list_empty(self):
        assert sort_list([]) == []

    def test_sort_list_duplicates(self):
        assert sort_list([2, 1, 2]) == [1, 2, 2]

# --- Tests for password_strength ---
class TestPasswordStrength:
    @pytest.mark.parametrize("pwd, expected", [
        ("Strong123", "strong"),
        ("weak", "weak"),        # Too short
        ("Short1", "weak"),      # Length < 8
        ("password123", "weak"), # No uppercase
        ("PASSWORD123", "strong"), # No lowercase (passes func logic but func only checks Uppers/Digits)
        ("OnlyLetters", "weak"), # No digits
        ("Abcdefg7", "strong"),  # Exactly 8, has Upper and Digit
    ])
    def test_password_variants(self, pwd, expected):
        assert password_strength(pwd) == expected
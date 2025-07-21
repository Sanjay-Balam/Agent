"""
DSA Knowledge Base - Contains templates, patterns, and examples for Data Structures and Algorithms.
This extends the system to support DSA topics while keeping Manim functionality intact.
"""

# Data Structure Templates
DATA_STRUCTURES = {
    "array": {
        "definition": "A collection of elements stored at contiguous memory locations",
        "code_template": """
# Array implementation
class Array:
    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.length = 0
    
    def get(self, index):
        if 0 <= index < self.length:
            return self.data[index]
        raise IndexError("Index out of bounds")
    
    def set(self, index, value):
        if 0 <= index < self.length:
            self.data[index] = value
        else:
            raise IndexError("Index out of bounds")
    
    def append(self, value):
        if self.length < self.size:
            self.data[self.length] = value
            self.length += 1
        else:
            raise OverflowError("Array is full")
""",
        "time_complexity": {
            "access": "O(1)",
            "search": "O(n)",
            "insertion": "O(n)",
            "deletion": "O(n)"
        },
        "space_complexity": "O(n)",
        "use_cases": ["Random access", "Cache-friendly operations", "Mathematical computations"]
    },
    
    "linked_list": {
        "definition": "A linear data structure where elements are stored in nodes, each pointing to the next",
        "code_template": """
# Linked List implementation
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, val):
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
""",
        "time_complexity": {
            "access": "O(n)",
            "search": "O(n)",
            "insertion": "O(1)",
            "deletion": "O(n)"
        },
        "space_complexity": "O(n)",
        "use_cases": ["Dynamic size", "Frequent insertions/deletions", "Stack/Queue implementation"]
    },
    
    "binary_tree": {
        "definition": "A hierarchical data structure with nodes having at most two children",
        "code_template": """
# Binary Tree implementation
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def inorder_traversal(self, node):
        result = []
        if node:
            result.extend(self.inorder_traversal(node.left))
            result.append(node.val)
            result.extend(self.inorder_traversal(node.right))
        return result
    
    def preorder_traversal(self, node):
        result = []
        if node:
            result.append(node.val)
            result.extend(self.preorder_traversal(node.left))
            result.extend(self.preorder_traversal(node.right))
        return result
    
    def postorder_traversal(self, node):
        result = []
        if node:
            result.extend(self.postorder_traversal(node.left))
            result.extend(self.postorder_traversal(node.right))
            result.append(node.val)
        return result
    
    def level_order_traversal(self):
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
""",
        "time_complexity": {
            "search": "O(n)",
            "insertion": "O(n)",
            "deletion": "O(n)",
            "traversal": "O(n)"
        },
        "space_complexity": "O(n)",
        "use_cases": ["Hierarchical data", "Expression parsing", "Decision trees"]
    },
    
    "binary_search_tree": {
        "definition": "A binary tree where left child < parent < right child",
        "code_template": """
# Binary Search Tree implementation
class BSTNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return BSTNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
    
    def delete(self, val):
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if not node:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            
            # Node with two children
            min_val = self._find_min(node.right)
            node.val = min_val
            node.right = self._delete_recursive(node.right, min_val)
        
        return node
    
    def _find_min(self, node):
        while node.left:
            node = node.left
        return node.val
""",
        "time_complexity": {
            "search": "O(log n) average, O(n) worst",
            "insertion": "O(log n) average, O(n) worst",
            "deletion": "O(log n) average, O(n) worst"
        },
        "space_complexity": "O(n)",
        "use_cases": ["Sorted data", "Range queries", "Efficient search operations"]
    },
    
    "hash_table": {
        "definition": "A data structure that maps keys to values using a hash function",
        "code_template": """
# Hash Table implementation
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        
        raise KeyError(f"Key '{key}' not found")
    
    def keys(self):
        all_keys = []
        for bucket in self.table:
            for key, _ in bucket:
                all_keys.append(key)
        return all_keys
""",
        "time_complexity": {
            "search": "O(1) average, O(n) worst",
            "insertion": "O(1) average, O(n) worst",
            "deletion": "O(1) average, O(n) worst"
        },
        "space_complexity": "O(n)",
        "use_cases": ["Fast lookups", "Caching", "Database indexing"]
    }
}

# Algorithm Templates
ALGORITHMS = {
    "bubble_sort": {
        "definition": "Simple comparison-based sorting algorithm",
        "code_template": """
def bubble_sort(arr):
    \"\"\"
    Bubble Sort Algorithm
    Time Complexity: O(n²)
    Space Complexity: O(1)
    \"\"\"
    n = len(arr)
    
    for i in range(n):
        # Flag to optimize - if no swaps, array is sorted
        swapped = False
        
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping happened, array is sorted
        if not swapped:
            break
    
    return arr

# Example usage
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr.copy())
print(f"Original: {arr}")
print(f"Sorted: {sorted_arr}")
""",
        "time_complexity": {
            "best": "O(n)",
            "average": "O(n²)",
            "worst": "O(n²)"
        },
        "space_complexity": "O(1)",
        "stable": True,
        "use_cases": ["Educational purposes", "Small datasets", "Nearly sorted data"]
    },
    
    "merge_sort": {
        "definition": "Divide-and-conquer sorting algorithm",
        "code_template": """
def merge_sort(arr):
    \"\"\"
    Merge Sort Algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    \"\"\"
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer
    return merge(left, right)

def merge(left, right):
    \"\"\"Merge two sorted arrays\"\"\"
    result = []
    i = j = 0
    
    # Merge elements in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(f"Original: {arr}")
print(f"Sorted: {sorted_arr}")
""",
        "time_complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)"
        },
        "space_complexity": "O(n)",
        "stable": True,
        "use_cases": ["Large datasets", "External sorting", "Stable sorting required"]
    },
    
    "quick_sort": {
        "definition": "Efficient divide-and-conquer sorting algorithm",
        "code_template": """
def quick_sort(arr, low=0, high=None):
    \"\"\"
    Quick Sort Algorithm
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n)
    \"\"\"
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition the array
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    \"\"\"Partition function using last element as pivot\"\"\"
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example usage
arr = [10, 7, 8, 9, 1, 5]
sorted_arr = quick_sort(arr.copy())
print(f"Original: {arr}")
print(f"Sorted: {sorted_arr}")
""",
        "time_complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²)"
        },
        "space_complexity": "O(log n)",
        "stable": False,
        "use_cases": ["General purpose sorting", "In-place sorting", "Good cache performance"]
    },
    
    "binary_search": {
        "definition": "Efficient search algorithm for sorted arrays",
        "code_template": """
def binary_search(arr, target):
    \"\"\"
    Binary Search Algorithm
    Time Complexity: O(log n)
    Space Complexity: O(1)
    \"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

def binary_search_recursive(arr, target, left=0, right=None):
    \"\"\"Recursive implementation of binary search\"\"\"
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
index = binary_search(arr, target)
print(f"Array: {arr}")
print(f"Target: {target}")
print(f"Index: {index}")
""",
        "time_complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)"
        },
        "space_complexity": "O(1) iterative, O(log n) recursive",
        "prerequisites": ["Array must be sorted"],
        "use_cases": ["Searching in sorted data", "Finding insertion point", "Range queries"]
    },
    
    "dynamic_programming_fibonacci": {
        "definition": "Optimized approach to solve overlapping subproblems",
        "code_template": """
def fibonacci_dp_memoization(n, memo={}):
    \"\"\"
    Fibonacci using Dynamic Programming (Memoization)
    Time Complexity: O(n)
    Space Complexity: O(n)
    \"\"\"
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_dp_memoization(n-1, memo) + fibonacci_dp_memoization(n-2, memo)
    return memo[n]

def fibonacci_dp_tabulation(n):
    \"\"\"
    Fibonacci using Dynamic Programming (Tabulation)
    Time Complexity: O(n)
    Space Complexity: O(n)
    \"\"\"
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def fibonacci_optimized(n):
    \"\"\"
    Space-optimized Fibonacci
    Time Complexity: O(n)
    Space Complexity: O(1)
    \"\"\"
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Example usage
n = 10
print(f"Fibonacci({n}) using memoization: {fibonacci_dp_memoization(n)}")
print(f"Fibonacci({n}) using tabulation: {fibonacci_dp_tabulation(n)}")
print(f"Fibonacci({n}) optimized: {fibonacci_optimized(n)}")
""",
        "time_complexity": {
            "naive": "O(2^n)",
            "dp": "O(n)",
            "optimized": "O(n)"
        },
        "space_complexity": {
            "memoization": "O(n)",
            "tabulation": "O(n)",
            "optimized": "O(1)"
        },
        "use_cases": ["Optimization problems", "Recursive problems with overlapping subproblems"]
    }
}

# DSA Problem Patterns
DSA_PATTERNS = {
    "two_pointers": {
        "description": "Use two pointers moving in same/opposite directions",
        "template": """
def two_sum_sorted(arr, target):
    \"\"\"Find two numbers that sum to target in sorted array\"\"\"
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
""",
        "use_cases": ["Array problems", "String problems", "Finding pairs/triplets"]
    },
    
    "sliding_window": {
        "description": "Maintain a window and slide it through the array",
        "template": """
def max_sum_subarray(arr, k):
    \"\"\"Find maximum sum of subarray of size k\"\"\"
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
""",
        "use_cases": ["Subarray problems", "String matching", "Fixed/variable window size"]
    },
    
    "fast_slow_pointers": {
        "description": "Use two pointers moving at different speeds",
        "template": """
def has_cycle(head):
    \"\"\"Detect cycle in linked list using Floyd's algorithm\"\"\"
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
""",
        "use_cases": ["Cycle detection", "Finding middle element", "Palindrome checking"]
    }
}

# Common DSA Questions and Solutions
COMMON_QUESTIONS = {
    "reverse_linked_list": {
        "question": "Reverse a singly linked list",
        "solution": """
def reverse_linked_list(head):
    \"\"\"
    Reverse a singly linked list
    Time: O(n), Space: O(1)
    \"\"\"
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev
""",
        "difficulty": "Easy",
        "topics": ["Linked List", "Pointers"]
    },
    
    "valid_parentheses": {
        "question": "Check if parentheses are valid and balanced",
        "solution": """
def is_valid(s):
    \"\"\"
    Check if parentheses are valid
    Time: O(n), Space: O(n)
    \"\"\"
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack
""",
        "difficulty": "Easy",
        "topics": ["Stack", "String"]
    },
    
    "two_sum": {
        "question": "Find two numbers that add up to target",
        "solution": """
def two_sum(nums, target):
    \"\"\"
    Two Sum problem
    Time: O(n), Space: O(n)
    \"\"\"
    seen = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []
""",
        "difficulty": "Easy",
        "topics": ["Array", "Hash Table"]
    }
}

def get_data_structure_info(ds_name):
    """Get information about a data structure."""
    return DATA_STRUCTURES.get(ds_name.lower())

def get_algorithm_info(algo_name):
    """Get information about an algorithm."""
    return ALGORITHMS.get(algo_name.lower())

def get_pattern_info(pattern_name):
    """Get information about a DSA pattern."""
    return DSA_PATTERNS.get(pattern_name.lower())

def get_question_solution(question_name):
    """Get solution for a common DSA question."""
    return COMMON_QUESTIONS.get(question_name.lower())

def list_available_topics():
    """List all available DSA topics."""
    return {
        "data_structures": list(DATA_STRUCTURES.keys()),
        "algorithms": list(ALGORITHMS.keys()),
        "patterns": list(DSA_PATTERNS.keys()),
        "questions": list(COMMON_QUESTIONS.keys())
    }
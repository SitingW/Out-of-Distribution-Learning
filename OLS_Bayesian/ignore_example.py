"""
Understanding Python Type Hints: Callable and Any
=================================================

Type hints help make your code more readable and enable better IDE support,
static analysis tools, and documentation. They don't affect runtime behavior.
"""

from typing import Callable, Any, List, Dict, Union, Optional

# 1. What is type hinting?
def without_type_hints(data, func):
    """Function without type hints - unclear what types are expected"""
    return func(data)

def with_type_hints(data: List[int], func: Callable[[List[int]], float]) -> float:
    """Function with type hints - clear what types are expected"""
    return func(data)

# 2. Understanding Callable
# Callable[input_types, return_type] describes a function signature

# Simple function that takes a list and returns a float
def simple_function(numbers: List[int]) -> float:
    return sum(numbers) / len(numbers)

# Function that takes two parameters
def two_param_function(numbers: List[int], multiplier: int) -> float:
    return sum(numbers) * multiplier

# Function that takes any number of arguments
def variable_args_function(*args) -> str:
    return f"Received {len(args)} arguments"

# 3. Different ways to specify Callable types
def demo_callable_types():
    """Demonstrates different Callable type specifications"""
    
    # Callable that takes a list of integers and returns a float
    func1: Callable[[List[int]], float] = simple_function
    
    # Callable that takes two parameters (list of int, int) and returns float
    func2: Callable[[List[int], int], float] = two_param_function
    
    # Callable that takes any arguments and returns a string
    func3: Callable[..., str] = variable_args_function  # ... means "any arguments"
    
    # Callable without specifying parameters (less specific)
    func4: Callable = simple_function  # Just says "this is a function"
    
    print("All function assignments successful!")
    return func1, func2, func3, func4

# 4. Understanding Any
# Any means "any type" - it's like disabling type checking for that parameter

def function_with_any(data: Any) -> Any:
    """Function that accepts any type and returns any type"""
    print(f"Received data of type: {type(data)}")
    return data

def process_mixed_data(items: List[Any]) -> None:
    """Function that processes a list containing any types"""
    for item in items:
        print(f"Item: {item}, Type: {type(item)}")

# 5. Why use Any?
# - When you genuinely don't know the type
# - When working with dynamic data
# - When migrating from untyped code gradually
# - When the type is too complex to specify

# 6. Better alternatives to Any when possible
def better_than_any_union(value: Union[int, str, float]) -> str:
    """Union types are more specific than Any"""
    return str(value)

def better_than_any_generic(value: Optional[str]) -> str:
    """Optional is better than Any for nullable values"""
    return value if value is not None else "default"

# 7. Real-world example from your inference function
def inference_function_typed(
    data: List[float], 
    method_func: Callable[[List[float]], float], 
    **kwargs: Any
) -> float:
    """
    Properly typed inference function
    
    Args:
        data: List of numbers to process
        method_func: Function that takes a list of floats and returns a float
        **kwargs: Additional keyword arguments of any type
    
    Returns:
        Float result from the method function
    """
    processed_data = [float(x) for x in data]  # Ensure data is floats
    result = method_func(processed_data, **kwargs)
    return result

# Example methods with proper type hints
def variance_method_typed(data: List[float], ddof: int = 0) -> float:
    """Calculate variance with proper type hints"""
    if len(data) < 2:
        return 0.0
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - ddof)
    return variance

def max_min_method_typed(data: List[float], absolute: bool = False) -> float:
    """Calculate max-min range with proper type hints"""
    if absolute:
        data = [abs(x) for x in data]
    return max(data) - min(data)

# 8. More complex Callable examples
def higher_order_function(
    data: List[int],
    transformer: Callable[[int], str],  # Function that converts int to string
    filter_func: Callable[[str], bool]  # Function that takes string, returns bool
) -> List[str]:
    """Example of multiple Callable parameters"""
    transformed = [transformer(x) for x in data]
    filtered = [x for x in transformed if filter_func(x)]
    return filtered

# 9. Demonstration
def main():
    print("=== Type Hints Demo ===")
    
    # Demo basic usage
    data = [1, 2, 3, 4, 5]
    
    # These work because the function signatures match the type hints
    result1 = inference_function_typed(data, variance_method_typed)
    print(f"Variance: {result1}")
    
    result2 = inference_function_typed(data, max_min_method_typed, absolute=True)
    print(f"Max-min (absolute): {result2}")
    
    # Demo Any usage
    print("\n=== Any Type Demo ===")
    function_with_any(42)
    function_with_any("hello")
    function_with_any([1, 2, 3])
    function_with_any({"key": "value"})
    
    mixed_list = [1, "hello", 3.14, True, [1, 2, 3]]
    process_mixed_data(mixed_list)
    
    # Demo higher-order functions
    print("\n=== Higher-Order Function Demo ===")
    numbers = [1, 2, 3, 4, 5]
    
    # Convert numbers to strings and filter out even-length strings
    result = higher_order_function(
        numbers,
        lambda x: f"number_{x}",  # int -> str
        lambda s: len(s) % 2 == 1  # str -> bool
    )
    print(f"Filtered strings: {result}")
    
    print("\n=== Key Benefits of Type Hints ===")
    print("1. Better IDE support (autocomplete, error detection)")
    print("2. Self-documenting code")
    print("3. Easier debugging and maintenance")
    print("4. Static analysis tools can catch errors")
    print("5. Better collaboration in teams")
    
    print("\n=== When to Use Each ===")
    print("Callable: When you need to pass functions as parameters")
    print("Any: When type is truly unknown or too complex to specify")
    print("Union: When you have a few specific possible types")
    print("Optional: When value might be None")

if __name__ == "__main__":
    main()
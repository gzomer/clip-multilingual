import os
import json
import pickle
import hashlib


def cache(folder, use_pickle=False):
    def decorator(original_func):
        def wrapper(*args, **kwargs):
            args_hash = hashlib.sha1(original_func.__name__.encode('utf-8')).hexdigest()
            if len(args):
                args_hash = hashlib.sha1(str(args).encode('utf-8')).hexdigest()

            # Get the cache file name
            cache_file = os.path.join(folder, args_hash)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            cache_file = os.path.join(folder, str(args_hash))
            # Check if the cache file exists
            if os.path.exists(cache_file):
                if use_pickle:
                    # Load the cache file
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(cache_file, 'r') as f:
                        data = f.read()
                        if data[0] == '{' or data[0] == '[':
                            return json.loads(data)
                        else:
                            return data

            # Call the original function
            result = original_func(*args, **kwargs)
            # Save the result to the cache file
            if use_pickle:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            else:
                if isinstance(result, object):
                    with open(cache_file, 'w') as f:
                        f.write(json.dumps(result, indent=4))
                else:
                    with open(cache_file, 'w') as f:
                        f.write(result)

            # Return the result
            return result

        return wrapper
    return decorator
from typing import Dict, Any, Optional, List

class DummyClass:
    def __init__(
            self, 
            fields: Optional[List[str]] = None, 
            fields_values: Optional[Dict[str, Any]] = None, 
            methods: Optional[List[str]] = None,
            methods_return_values: Optional[Dict[str, Any]] = None,
        ):
        '''
        Initializes a DummyClass. 
        When asked for a field from `fields` it returns the corresponding `field_value`.
        When called for a method from `methods` it returns the corresponding `methods_return_values`.
        '''
        if fields is None:
            fields = []
        if fields_values is None:
            fields_values = {}
        if methods is None:
            methods = []
        if methods_return_values is None:
            methods_return_values = {}

        for field in fields:
            setattr(self, field, fields_values.get(field, None))
        for method in methods:
            setattr(self, method, lambda *args, **kwargs: methods_return_values.get(method, None))


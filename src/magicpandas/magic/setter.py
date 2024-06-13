from __future__ import annotations


class __setter__:
    def __get__(self, instance: object, owner):
        if instance is None:
            return self
        key = self.__name__
        if key in instance.__dict__:
            return (
                instance.__dict__
                [key]
                .__get__(instance, owner)
            )
        return self

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(self, instance: object, value):
        instance.__dict__[self.__name__] = value

    def __call__(self, value):
        return value

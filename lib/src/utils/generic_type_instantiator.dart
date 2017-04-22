import 'dart:mirrors';

class Instantiator {
  static createInstance(Type type, [Symbol constructor = const Symbol(""), List arguments = const [],
    Map<Symbol, dynamic> namedArguments]) {

    if (type == null) {
      throw new ArgumentError("type: $type");
    }

    TypeMirror typeMirror = reflectType(type);

    if (typeMirror is ClassMirror) {
      return typeMirror.newInstance(constructor, arguments, namedArguments).reflectee;
    }

    throw new ArgumentError("Cannot create the instance of the type '$type'.");
  }
}
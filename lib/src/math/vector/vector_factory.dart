import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/math/vector/regular_vector.dart';
import 'package:dart_ml/src/math/vector/typed_vector.dart';
import 'package:dart_ml/src/utils/generic_type_instantiator.dart';

class VectorFactory {
  static VectorInterface createFilled(Type type, int dimension, double value) {
    VectorInterface vector = Instantiator.createInstance(type, const Symbol(''), [0]);

    if (vector is RegularVector) {
      return new RegularVector.filled(dimension, value);
    }

    if (vector is TypedVector) {
      return new TypedVector.filled(dimension, value);
    }

    throw new Exception("Unsupported type $type");
  }

  static VectorInterface createFrom(Type type, Iterable<double> source) {
    VectorInterface vector = Instantiator.createInstance(type, const Symbol(''), [0]);

    if (vector is RegularVector) {
      return new RegularVector.from(source);
    }

    if (vector is TypedVector) {
      return new TypedVector.from(source);
    }

    throw new Exception("Unsupported type $type");
  }
}
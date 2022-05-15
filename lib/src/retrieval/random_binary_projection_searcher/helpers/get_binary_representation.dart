import 'package:ml_linalg/matrix.dart';

Matrix getBinaryRepresentation(Matrix data, Matrix randomVectors) {
  return (data * randomVectors).mapElements((element) => element >= 0 ? 1 : 0);
}

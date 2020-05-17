import 'package:ml_linalg/linalg.dart';

const inverseLogitLinkFunctionFloat32Token = 'inverseLogitLinkFunctionFloat32';
const inverseLogitLinkFunctionFloat64Token = 'inverseLogitLinkFunctionFloat64';
const softmaxLinkFunctionFloat32Token = 'softmaxLinkFunctionFloat32';
const softmaxLinkFunctionFloat64Token = 'softmaxLinkFunctionFloat64';

const dTypeToInverseLogitLinkFunctionToken = {
  DType.float32: inverseLogitLinkFunctionFloat32Token,
  DType.float64: inverseLogitLinkFunctionFloat64Token,
};

const dTypeToSoftmaxLinkFunctionToken = {
  DType.float32: softmaxLinkFunctionFloat32Token,
  DType.float64: softmaxLinkFunctionFloat64Token,
};

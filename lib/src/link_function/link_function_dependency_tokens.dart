import 'package:ml_linalg/linalg.dart';

const float32InverseLogitLinkFunctionToken = 'float32InverseLogitLinkFunction';
const float64InverseLogitLinkFunctionToken = 'float64InverseLogitLinkFunction';
const float32SoftmaxLinkFunctionToken = 'float32SoftmaxLinkFunction';
const float64SoftmaxLinkFunctionToken = 'float64SoftmaxLinkFunction';

const dTypeToInverseLogitLinkFunctionToken = {
  DType.float32: float32InverseLogitLinkFunctionToken,
  DType.float64: float64InverseLogitLinkFunctionToken,
};

const dTypeToSoftmaxLinkFunctionToken = {
  DType.float32: float32SoftmaxLinkFunctionToken,
  DType.float64: float64SoftmaxLinkFunctionToken,
};

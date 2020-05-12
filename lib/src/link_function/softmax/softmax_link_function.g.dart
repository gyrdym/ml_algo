// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'softmax_link_function.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SoftmaxLinkFunction _$SoftmaxLinkFunctionFromJson(Map<String, dynamic> json) {
  return $checkedNew('SoftmaxLinkFunction', json, () {
    $checkKeys(json, allowedKeys: const ['DT']);
    final val = SoftmaxLinkFunction(
      $checkedConvert(json, 'DT', (v) => fromDTypeJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {'dtype': 'DT'});
}

Map<String, dynamic> _$SoftmaxLinkFunctionToJson(
        SoftmaxLinkFunction instance) =>
    <String, dynamic>{
      'DT': dTypeToJson(instance.dtype),
    };

// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'inverse_logit_link_function.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

InverseLogitLinkFunction _$InverseLogitLinkFunctionFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('InverseLogitLinkFunction', json, () {
    $checkKeys(json, allowedKeys: const ['DT']);
    final val = InverseLogitLinkFunction(
      $checkedConvert(json, 'DT', (v) => fromDTypeJson(v as String)),
    );
    return val;
  }, fieldKeyMap: const {'dtype': 'DT'});
}

Map<String, dynamic> _$InverseLogitLinkFunctionToJson(
        InverseLogitLinkFunction instance) =>
    <String, dynamic>{
      'DT': dTypeToJson(instance.dtype),
    };

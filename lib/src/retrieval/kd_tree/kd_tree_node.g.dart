// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeNode _$KDTreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('KDTreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['V', 'I', 'L', 'R', 'S']);
    final val = KDTreeNode(
      value: $checkedConvert(json, 'V',
          (v) => v == null ? null : Vector.fromJson(v as Map<String, dynamic>)),
      index: $checkedConvert(json, 'I', (v) => v as int?),
      left: $checkedConvert(
          json,
          'L',
          (v) => v == null
              ? null
              : KDTreeNode.fromJson(v as Map<String, dynamic>)),
      right: $checkedConvert(
          json,
          'R',
          (v) => v == null
              ? null
              : KDTreeNode.fromJson(v as Map<String, dynamic>)),
      samples: $checkedConvert(json, 'S',
          (v) => v == null ? null : Matrix.fromJson(v as Map<String, dynamic>)),
    );
    return val;
  }, fieldKeyMap: const {
    'value': 'V',
    'index': 'I',
    'left': 'L',
    'right': 'R',
    'samples': 'S'
  });
}

Map<String, dynamic> _$KDTreeNodeToJson(KDTreeNode instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('V', instance.value?.toJson());
  writeNotNull('I', instance.index);
  writeNotNull('L', instance.left?.toJson());
  writeNotNull('R', instance.right?.toJson());
  writeNotNull('S', instance.samples?.toJson());
  return val;
}

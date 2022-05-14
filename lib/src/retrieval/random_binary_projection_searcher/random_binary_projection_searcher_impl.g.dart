// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'random_binary_projection_searcher_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RandomBinaryProjectionSearcherImpl _$RandomBinaryProjectionSearcherImplFromJson(
    Map<String, dynamic> json) {
  return $checkedNew('RandomBinaryProjectionSearcherImpl', json, () {
    $checkKeys(json, allowedKeys: const ['S', 'D', 'H', 'P', 'R', 'B', r'$V']);
    final val = RandomBinaryProjectionSearcherImpl(
      $checkedConvert(
          json, 'H', (v) => (v as List<dynamic>).map((e) => e as String)),
      $checkedConvert(
          json, 'P', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'D', (v) => v as int),
      seed: $checkedConvert(json, 'S', (v) => v as int?),
      schemaVersion: $checkedConvert(json, r'$V', (v) => v as int),
    );
    $checkedConvert(json, 'R',
        (v) => val.randomVectors = Matrix.fromJson(v as Map<String, dynamic>));
    $checkedConvert(
        json,
        'B',
        (v) => val.bins = (v as Map<String, dynamic>).map(
              (k, e) => MapEntry(int.parse(k),
                  (e as List<dynamic>).map((e) => e as int).toList()),
            ));
    return val;
  }, fieldKeyMap: const {
    'header': 'H',
    'points': 'P',
    'digitCapacity': 'D',
    'seed': 'S',
    'schemaVersion': r'$V',
    'randomVectors': 'R',
    'bins': 'B'
  });
}

Map<String, dynamic> _$RandomBinaryProjectionSearcherImplToJson(
    RandomBinaryProjectionSearcherImpl instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('S', instance.seed);
  val['D'] = instance.digitCapacity;
  val['H'] = instance.header.toList();
  val['P'] = instance.points.toJson();
  val['R'] = instance.randomVectors.toJson();
  val['B'] = instance.bins.map((k, e) => MapEntry(k.toString(), e));
  val[r'$V'] = instance.schemaVersion;
  return val;
}

import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_json_keys.dart';
import 'package:test/test.dart';

void main() {
  group('TreeLeafLabel', () {
    test('should throw an exception if probability is less than 0', () {
      final labelValue = 5;
      final probability = -0.001;

      expect(() => TreeLeafLabel(labelValue, probability: probability),
          throwsRangeError);
    });

    test('should allow probability to be negative with small penalty', () {
      final labelValue = 5;
      final probability = -1e-5;
      final label = TreeLeafLabel(labelValue, probability: probability);

      expect(label.probability, probability);
    });

    test('should throw an exception if probability is greater than 1', () {
      final labelValue = -1e5;
      final probability = 1.001;

      expect(() => TreeLeafLabel(labelValue, probability: probability),
          throwsRangeError);
    });

    test('should allow probability to be greater than 1 with small penalty',
        () {
      final labelValue = 5;
      final probability = 1 + 1e-5;
      final label = TreeLeafLabel(labelValue, probability: probability);

      expect(label.probability, probability);
    });

    test('should allow probability to be 0', () {
      final labelValue = 5;
      final probability = 0;
      final label = TreeLeafLabel(labelValue, probability: probability);

      expect(label.probability, probability);
    });

    test('should allow probability to be 1', () {
      final labelValue = 5;
      final probability = 1;
      final label = TreeLeafLabel(labelValue, probability: probability);

      expect(label.probability, probability);
    });

    test('should store value', () {
      final labelValue = -1e5;
      final probability = 0.3;
      final leafLabel = TreeLeafLabel(labelValue, probability: probability);

      expect(leafLabel.value, labelValue);
    });

    test('should store probability', () {
      final labelValue = -1e5;
      final probability = 0.3;
      final leafLabel = TreeLeafLabel(labelValue, probability: probability);

      expect(leafLabel.probability, probability);
    });

    test('should serialize (probability value is null)', () {
      final labelValue = 1000;
      final probability = 0.5;
      final leafLabel = TreeLeafLabel(labelValue, probability: probability);
      final serialized = leafLabel.toJson();

      expect(
          serialized,
          equals({
            leafLabelValueJsonKey: labelValue,
            leafLabelProbabilityJsonKey: probability,
          }));
    });

    test('should serialize (probability value is not null)', () {
      final labelValue = -1000;
      final probability = 0.7;
      final leafLabel = TreeLeafLabel(labelValue, probability: probability);
      final serialized = leafLabel.toJson();

      expect(
          serialized,
          equals({
            leafLabelValueJsonKey: labelValue,
            leafLabelProbabilityJsonKey: probability,
          }));
    });

    test('should restore from json (probability value is null)', () {
      final labelValue = 12345;
      final probability = 0.7;
      final json = {
        leafLabelValueJsonKey: labelValue,
        leafLabelProbabilityJsonKey: probability,
      };
      final restored = TreeLeafLabel.fromJson(json);

      expect(restored.value, labelValue);
      expect(restored.probability, probability);
    });

    test('should restore from json (probability value is not null)', () {
      final labelValue = 12345;
      final probability = 0.75;
      final json = {
        leafLabelValueJsonKey: labelValue,
        leafLabelProbabilityJsonKey: probability,
      };
      final restored = TreeLeafLabel.fromJson(json);

      expect(restored.value, labelValue);
      expect(restored.probability, probability);
    });

    test('should return a proper schema version', () {
      final leafLabel = TreeLeafLabel(1, probability: 0.5);

      expect(leafLabel.schemaVersion, 1);
    });
  });
}

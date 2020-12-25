import 'dart:io';

abstract class Serializable {
  /// Returns a json-serializable map
  Map<String, dynamic> toJson();

  /// Saves a json-serializable map into a newly created file with the path
  /// [filePath]
  Future<File> saveAsJson(String filePath);

  /// Contains a version of the current json schema
  int get schemaVersion;
}

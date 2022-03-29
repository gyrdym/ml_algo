class KDTreeNeighbour {
  KDTreeNeighbour(this.index, this.distance);

  final int index;
  final num distance;

  @override
  bool operator ==(Object other) {
    if (other is KDTreeNeighbour) {
      return index == other.index;
    }

    return false;
  }

  @override
  String toString() => '(Index: $index, Distance: $distance)';
}

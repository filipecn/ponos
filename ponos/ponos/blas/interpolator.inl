template<class P, typename T>
RBFInterpolator<P, T>::RBFInterpolator(Kernel <T> *kernel) : kernel_(kernel) {}

template<class P, typename T>
RBFInterpolator<P, T>::~RBFInterpolator() {}

template<class P, typename T>
T RBFInterpolator<P, T>::interpolateAt(P target, const std::vector<P> &points, const std::vector<T> &values) {
  uint n = points.size();
  auto dim = P::dimension();
  LinearSystem<DenseMatrix<T>, DenseVector<T>> system;
  system.resize(n + dim + 1, n + dim + 1);
  GaussJordanSolver<LinearSystem<DenseMatrix<T>, DenseVector<T>>> solver;

  for (uint i = 0; i < n; i++) {
    for (uint j = i; j < n; j++)
      system.A(i, j) = system.A(j, i) = (*kernel_)(distance(points[i], points[j]));
    system.A(i, n) = system.A(n, i) = 1;
    for (uint d = 1; d <= dim; d++)
      system.A(i, n + d) = system.A(n + d, i) = (points[i])[d - 1];
    system.b[i] = values[i];
  }
  solver.solve(&system);

  double sum = 0;
  for (uint i = 0; i < n; i++)
    sum += system.b[i] * (*kernel_)(distance(target, points[i]));
  sum += system.b[n];
  for (uint d = 1; d <= dim; d++)
    sum += system.b[n + d] * target[d - 1];
  return sum;
}

template<class P, typename T>
std::vector<double> RBFInterpolator<P, T>::weights(P target, const std::vector<P> &points) const {
  UNUSED_VARIABLE(target);
  UNUSED_VARIABLE(points);
  return std::vector<double>();
}

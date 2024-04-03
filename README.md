# adversarial-invariant

    What is invariant under the PGD advesarial attack??

----

```
⚪ representation distillation
let f be the classifier for the downstream task
  f(x) -> y
the attacker max. loss(y, y') to obtain the noise distribution dx ~ F
  f(x+dx) -> y' != y
there must exist something invariant against F, which could be obtained through g:
  g(x) -> z
  g(x+dx) -> z' ~= z
[Q1]: what would z look like, if represented as a grey image sized smaller than x?

⚪ model distillation
then we train another classfier for the original task, with z as input
  h(z) -> y
again the attacker will attack over h to get new noises dz ~ H
  h(z+dz) -> y' != y
[Q2]: will this h be harder to attack than the original y?

possible issues:
- it is a kind of layer-wise PGD-noise consistency restriction
- the implementation of g can be a parametricalized guidedFilter
- the implementation of h is just a small size of f?
- g may face the modal collapse problem, so that z is not that distinguishable to h
```

----
by Armit
2023/12/22

subroutine getConstants(n, k, q, Voc, V0, AT, Isat)

  implicit none

  !Output
  double precision, intent(out) ::  n, k, q, Voc, V0, AT, Isat

  n = 1.35
  k = 1.38065e-23
  q = 1.60218e-19
  Voc = 2.68/3.0
  V0 = -0.6
  AT = 2.66e-3
  Isat = 2.809e-12

end subroutine getConstants




subroutine computeVcurve(nT, nA, nI, T, A, I, V)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) nT, nA, nI, T, A, I
  !f2py intent(out) V
  !f2py depend(nT) T
  !f2py depend(nA) A
  !f2py depend(nI) I
  !f2py depend(nT,nA,nI) V

  !Input
  integer, intent(in) ::  nT, nA, nI
  double precision, intent(in) ::  T(nT), A(nA), I(nI)

  !Output
  double precision, intent(out) ::  V(nT,nA,nI)

  !Working
  double precision n, k, q, Voc, V0, AT, Isat
  double precision Isc, Vt
  double precision arg, den
  integer iT, iA, iI

  call getConstants(n, k, q, Voc, V0, AT, Isat)

  do iI=1,nI
     do iA=1,nA
        do iT=1,nT
           Isc = 0.453*A(iA)/AT
           Vt = n*k*T(iT)/q
           arg = (Isc + Isat - I(iI)) / Isat
           den = I(iI) - Isc - V0*Isat/Vt
           if (I(iI) .le. Isc) then
              V(iT,iA,iI) = Vt*log(arg)
           else
              V(iT,iA,iI) = V0**2*Isat/Vt/den + V0
           end if
        end do
     end do
  end do


end subroutine computeVcurve

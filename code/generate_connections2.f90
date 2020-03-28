! File generate_connections.f90
SUBROUTINE others(conmat,n,maxcother,randsinput,randsinputsorted,kk,th,connectionothermax)

IMPLICIT NONE
!Pythonic array indices, from 0 to n-1.

integer (kind=4), dimension(0:n-1), intent(in) :: randsinput, randsinputsorted
integer (kind=4), dimension(0:n-1, 0:maxcother-1), intent(inout) :: conmat
integer (kind=4), dimension(0:n-1), intent(out) :: connectionothermax
integer (kind=4), dimension(0:maxcother) :: tpind
integer (kind=4), dimension(0:maxcother-1) :: cons
integer, intent(in) :: n,maxcother
integer :: i,j,pn,iad,pnj,diff,indmax,indmin,ind_cind,suma
real (kind=4) :: pnj_real
real (kind=8) :: rnd
real, intent(in) :: kk,th
logical :: first
REAL     :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0,   &
                      vsmall = TINY(1.0), vlarge = HUGE(1.0)

INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12, 60)
real (kind=4) :: random_gamma

!f2py intent(in) :: n,maxcother,randsinput,randsinputsorted,k,th
!f2py intent(in,out) :: conmat
!f2py intent(out) :: connectionothermax
!f2py intent(hide) :: n,maxcother

!write(6,*) kk,th

cons(:) = 0

first = .true.

! compute number of nodes with j contacts
do i = 0,n-1
  j = randsinput(i)
  cons(j) = cons(j) + 1
enddo
!write(6,*) cons
! compute max,min indices of nodes with j contacts in randsinputsorted
tpind(:) = 0
suma = 0
do j = 0,maxcother-1
  suma = suma + cons(j)
  tpind(j+1) = suma
enddo

!OPEN(UNIT=12, FILE="numbers.txt", ACTION="write", STATUS="replace")
do i = 0,n-1
  pn = randsinput(i) ! number of connections of node i
  
  ! randomly draw the connected nodes based on discrete probability distribution
  do j = 0,pn-1
    ! how many connections has the neigbouring node?
    pnj_real = (random_gamma(kk, first)-1.)*th
    first = .false.
    pnj = NINT(pnj_real)

    !WRITE(12,FMT="(F7.2)") pnj_real

    
    ! dont allow pnj larger than maxcother
    if ( pnj > maxcother - 1 ) then
      pnj = maxcother - 1
    endif

    ! the indices that have "pnj" connections
    indmax = tpind(pnj+1)
    indmin = tpind(pnj)    

    diff = indmax - indmin
    if ( diff /= 0 .and. indmax < n ) then
      ! uniform sampling over cind, which includes indices "pnj" connections
      call random_number(rnd)

      ind_cind = FLOOR(diff*rnd)
      iad = randsinputsorted(indmin+ind_cind)

      ! wire the connections
      if ( connectionothermax(i) < maxcother .and. connectionothermax(iad) < maxcother ) then
        conmat(i,connectionothermax(i)) = iad
        conmat(iad,connectionothermax(iad)) = i
        connectionothermax(i) = connectionothermax(i) + 1
        connectionothermax(iad) = connectionothermax(iad) + 1
      endif
    endif
  end do
end do
! CLOSE(UNIT=12)

END SUBROUTINE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine household(conmat,n,maxfother,connectionfamilymax)
implicit none

integer (kind=4), dimension(0:n-1, 0:maxfother-1), intent(inout) :: conmat
integer (kind=4), dimension(0:n-1), intent(out) :: connectionfamilymax
integer, intent(in) :: n,maxfother
integer :: h1,h2,h3,h4,h5,h6,h7,h8,ec_size,ec_centers,pp_center,group_size,gp_center
integer :: i,j,k,l,ps,en,a,b

!f2py intent(in) :: n,maxfother
!f2py intent(in,out) :: conmat
!f2py intent(out) :: connectionfamilymax
!f2py intent(hide) :: n,maxfother


connectionfamilymax(:) = 0


! households in Slovenia: https://www.stat.si/StatWeb/News/Index/7725
h1 = 269898 ! 1 person
h2 = 209573 ! 2 person
h3 = 152959 ! 3 person
h4 = 122195 ! 4 person
h5 =  43327 ! 5 person
h6 =  17398 ! 6 person
h7 =   6073 ! 7 person
h8 =   3195 ! 8 person

! elderly care
ec_size = 20000  ! persons in elderly care centers
ec_centers = 100 ! elderly care centers (100 for simplicity, actually 102)
pp_center = 200  ! people per center
group_size = 25  ! number of poeple in one group
gp_center = 8    ! #groups per center

! generate h1
i = h1
write(6,*) i

! generate h2
ps = 2
en = i + ps*h2

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h3
ps = 3
en = i + ps*h3

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h4
ps = 4
en = i + ps*h4

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h5
ps = 5
en = i + ps*h5

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h6
ps = 6
en = i + ps*h6

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h7
ps = 7
en = i + ps*h7

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i

! generate h8
ps = 8
en = i + ps*h8

do while (i < en)
  do j = 0,ps-1
    l = 0
    do k = 0,ps-1
      if (j /= k) then
         conmat(i+j,l) = i+k
         connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
         l = l + 1
      endif
    enddo
  enddo
  i = i + ps
enddo
write(6,*) i


! elderly centers - groups of 25 in a center of 200
en = i + ec_size
do while (i < en)
  do a = 0,ec_centers-1
    do b = 0,gp_center-1
        do j = 0,group_size-1
          l = 0
          do k = 0,group_size-1
            if (j /= k) then
              conmat(i+j,l) = i+k
              connectionfamilymax(i+j) = connectionfamilymax(i+j) + 1
              l = l +1 
            endif
          enddo
        enddo        
      i = i + group_size 
    enddo
  enddo
enddo
write(6,*) i
end subroutine


FUNCTION random_normal() RESULT(fn_val)

! Adapted from the following Fortran 77 code
!      ALGORITHM 712, COLLECTED ALGORITHMS FROM ACM.
!      THIS WORK PUBLISHED IN TRANSACTIONS ON MATHEMATICAL SOFTWARE,
!      VOL. 18, NO. 4, DECEMBER, 1992, PP. 434-435.

!  The function random_normal() returns a normally distributed pseudo-random
!  number with zero mean and unit variance.

!  The algorithm uses the ratio of uniforms method of A.J. Kinderman
!  and J.F. Monahan augmented with quadratic bounding curves.

REAL :: fn_val

!     Local variables
REAL     :: s = 0.449871, t = -0.386595, a = 0.19600, b = 0.25472,    &
            r1 = 0.27597, r2 = 0.27846, u, v, x, y, q

!     Generate P = (u,v) uniform in rectangle enclosing acceptance region

DO
  CALL RANDOM_NUMBER(u)
  CALL RANDOM_NUMBER(v)
  v = 1.7156 * (v - half)

!     Evaluate the quadratic form
  x = u - s
  y = ABS(v) - t
  q = x**2 + y*(a*y - b*x)

!     Accept P if inside inner ellipse
  IF (q < r1) EXIT
!     Reject P if outside outer ellipse
  IF (q > r2) CYCLE
!     Reject P if outside acceptance region
  IF (v**2 < -4.0*LOG(u)*u**2) EXIT
END DO

!     Return ratio of P's coordinates as the normal deviate
fn_val = v/u
RETURN

END FUNCTION random_normal


FUNCTION random_gamma(s, first) RESULT(fn_val)

! Adapted from Fortran 77 code from the book:
!     Dagpunar, J. 'Principles of random variate generation'
!     Clarendon Press, Oxford, 1988.   ISBN 0-19-852202-9

!     FUNCTION GENERATES A RANDOM GAMMA VARIATE.
!     CALLS EITHER random_gamma1 (S > 1.0)
!     OR random_exponential (S = 1.0)
!     OR random_gamma2 (S < 1.0).

!     S = SHAPE PARAMETER OF DISTRIBUTION (0 < REAL).

REAL, INTENT(IN)    :: s
LOGICAL, INTENT(IN) :: first
REAL                :: fn_val
REAL :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0

IF (s <= zero) THEN
  WRITE(*, *) 'SHAPE PARAMETER VALUE MUST BE POSITIVE'
  STOP
END IF

IF (s > one) THEN
  fn_val = random_gamma1(s, first)
ELSE IF (s < one) THEN
  fn_val = random_gamma2(s, first)
ELSE
  fn_val = random_exponential()
END IF

RETURN
END FUNCTION random_gamma



FUNCTION random_gamma1(s, first) RESULT(fn_val)

! Uses the algorithm in
! Marsaglia, G. and Tsang, W.W. (2000) `A simple method for generating
! gamma variables', Trans. om Math. Software (TOMS), vol.26(3), pp.363-372.

! Generates a random gamma deviate for shape parameter s >= 1.

REAL, INTENT(IN)    :: s
LOGICAL, INTENT(IN) :: first
REAL                :: fn_val
REAL :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0
! Local variables
REAL, SAVE  :: c, d
REAL        :: u, v, x

IF (first) THEN
  d = s - one/3.
  c = one/SQRT(9.0*d)
END IF

! Start of main loop
DO

! Generate v = (1+cx)^3 where x is random normal; repeat if v <= 0.

  DO
    x = random_normal()
    v = (one + c*x)**3
    IF (v > zero) EXIT
  END DO

! Generate uniform variable U

  CALL RANDOM_NUMBER(u)
  IF (u < one - 0.0331*x**4) THEN
    fn_val = d*v
    EXIT
  ELSE IF (LOG(u) < half*x**2 + d*(one - v + LOG(v))) THEN
    fn_val = d*v
    EXIT
  END IF
END DO

RETURN
END FUNCTION random_gamma1





FUNCTION random_gamma2(s, first) RESULT(fn_val)

! Adapted from Fortran 77 code from the book:
!     Dagpunar, J. 'Principles of random variate generation'
!     Clarendon Press, Oxford, 1988.   ISBN 0-19-852202-9

! FUNCTION GENERATES A RANDOM VARIATE IN [0,INFINITY) FROM
! A GAMMA DISTRIBUTION WITH DENSITY PROPORTIONAL TO
! GAMMA2**(S-1) * EXP(-GAMMA2),
! USING A SWITCHING METHOD.

!    S = SHAPE PARAMETER OF DISTRIBUTION
!          (REAL < 1.0)

REAL, INTENT(IN)    :: s
LOGICAL, INTENT(IN) :: first
REAL                :: fn_val

!     Local variables
REAL       :: r, x, w
REAL, SAVE :: a, p, c, uf, vr, d
REAL :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0

IF (s <= zero .OR. s >= one) THEN
  WRITE(*, *) 'SHAPE PARAMETER VALUE OUTSIDE PERMITTED RANGE'
  STOP
END IF

IF (first) THEN                        ! Initialization, if necessary
  a = one - s
  p = a/(a + s*EXP(-a))
  IF (s < vsmall) THEN
    WRITE(*, *) 'SHAPE PARAMETER VALUE TOO SMALL'
    STOP
  END IF
  c = one/s
  uf = p*(vsmall/a)**s
  vr = one - vsmall
  d = a*LOG(a)
END IF

DO
  CALL RANDOM_NUMBER(r)
  IF (r >= vr) THEN
    CYCLE
  ELSE IF (r > p) THEN
    x = a - LOG((one - r)/(one - p))
    w = a*LOG(x)-d
  ELSE IF (r > uf) THEN
    x = a*(r/p)**c
    w = x
  ELSE
    fn_val = zero
    RETURN
  END IF

  CALL RANDOM_NUMBER(r)
  IF (one-r <= w .AND. r > zero) THEN
    IF (r*(w + one) >= one) CYCLE
    IF (-LOG(r) <= w) CYCLE
  END IF
  EXIT
END DO

fn_val = x
RETURN

END FUNCTION random_gamma2

FUNCTION random_exponential() RESULT(fn_val)

! Adapted from Fortran 77 code from the book:
!     Dagpunar, J. 'Principles of random variate generation'
!     Clarendon Press, Oxford, 1988.   ISBN 0-19-852202-9

! FUNCTION GENERATES A RANDOM VARIATE IN [0,INFINITY) FROM
! A NEGATIVE EXPONENTIAL DlSTRIBUTION WlTH DENSITY PROPORTIONAL
! TO EXP(-random_exponential), USING INVERSION.

REAL  :: fn_val

!     Local variable
REAL  :: r
REAL :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0

DO
  CALL RANDOM_NUMBER(r)
  IF (r > zero) EXIT
END DO

fn_val = -LOG(r)
RETURN

END FUNCTION random_exponential

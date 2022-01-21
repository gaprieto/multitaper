program test_dpss

!********************************************************************

   implicit none

!  Important variables

   integer, parameter :: npts=101, nev=7
   real(8) :: fw
   real(8), dimension(npts,nev) :: v
   real(8), dimension(nev) :: lambda, theta

!********************************************************************

    fw = 4.d0
    call dpss(npts, fw, nev, v, lambda, theta)

end program

subroutine dpss(npts, fw, nev, v, lambda, theta)

!
!  Modified:
!	German Prieto
!	November 2004
!
!  Calculation of the Discrete Prolate Spheroidal Sequences, and 
!  the correspondent eigenvalues. Also, the (1 - eigenvalue) terms
!  are calculated. 
!
!  Slepian, D.     1978  Bell Sys Tech J v57 n5 1371-1430
!  Thomson, D. J.  1982  Proc IEEE v70 n9 1055-1096
!
!  Input:
!    npts    	the number of points in the series
!    fw    	the time-bandwidth product (number of Rayleigh bins)
!    nev   	the desired number of tapers
!
!  Output:
!    v     	the eigenvectors (tapers) are returned in v(npts,nev)
!    lambda 	the eigenvalues of the v's
!    theta	the 1-lambda values. The energy outside the bandwidth
!
!  The tapers are the eigenvectors of the tridiagonal matrix sigma(i,j)
!  [see Slepian(1978) eq 14 and 25.] They are also the eigenvectors of
!  the Toeplitz matrix eq. 18. We solve the tridiagonal system in
!  rst_eig (real symmetric tridiagonal solver) for the tapers and use 
!  them in the integral equation in the frequency domain 
!  (dpss_ev subroutine) to get the eigenvalues more accurately, 
!  by performing Chebychev Gaussian Quadrature following Thomson's codes.
!
!  First, we create the main and off-diagonal vectors of the 
!  tridiagonal matrix. We compute separetely the even and odd tapers, 
!  by calling rst_eig from LAPACK.
!  We, refine the eigenvalues, by computing the inner bandwidth 
!  energy in the frequency domain (eq. 2.6 Thomson). Also the "leakage"
!  (1 - eigenvalue) is estimated, independenly if necesary. 
!
!  calls rst_eig, dpss_ev
!

!********************************************************************

   implicit none

!  Important variables

   integer, intent(in) :: npts, nev
   real(8), intent(in) :: fw 

   real(8), intent(out), dimension(nev) :: lambda, theta
   real(8), intent(out), dimension(npts,nev) :: v

!  Parameters

   real(8), parameter :: pi=3.141592653589793d0, r2=1.4142135623731d0

   integer :: ntot, lh, nr, k, kr, k2, n, i, ierr, nx
   
   real(8) :: atol, sn, lb, ub

!  Working variables

   integer :: neven, nodd
   integer, dimension(nev) :: ind
   
   real(8) :: bw, hn, com
   real(8), dimension(nev) :: eigval
   real(8), dimension(:), allocatable :: fv1, fv2
   real(8), dimension(:,:), allocatable :: v2

!********************************************************************

   if (npts < 2) then
      return
   endif

   nx=mod(npts,2)
   lh=(npts/2)+nx

   nodd  = nev/2
   neven = nev - nodd

   print *, 'npts lh nx ', npts, lh, nx
   print *, '101/2, 100/2', npts/2, (npts+1)/2
   print *, nev, nodd, neven

   bw = fw/dble(npts)
   com = cos(2.d0*pi*bw)

   hn = dble(npts-1)/2.d0

   print *, bw,com,hn

   if (allocated(fv1)) then
      deallocate(fv1)
   endif 
   if (allocated(fv2)) then
      deallocate(fv2)
   endif 

   allocate(fv1(lh))
   allocate(fv2(lh))

!
!  Perform symmetry reduction to half size
!

   do i=1,lh
      n = i-1
!     Main diagonal
      fv1(i)    = com*(hn - dble(n))**2.d0
!     sub diagonal
      fv2(i)    = dble(n*(npts-n))/2.d0

   enddo
  
   if (nx.eq.0) then
      fv1(lh)   = com* (hn - dble(lh-1))**2.d0 + dble(lh*(npts-lh))/2.d0
   else
      fv2(lh)   = r2*fv2(lh)
   endif
 
   do i=1,lh
      print *, fv1(i), fv2(i)
   enddo

   return
 
!
!  Do the even tapers
!
   
   if (allocated(v2)) then
      deallocate(v2)
   endif   
   allocate(v2(lh,neven))
   
   call rst_eig(lh,fv1,fv2(2:lh),neven,eigval,v2)

   if (nx==1) then
      do k = 1,neven
         v2(lh,k) = r2*v2(lh,k)
      enddo
   endif

   do k = 1,neven
      kr = k
      k2 = 2*k - 1

      theta(k2) = eigval(kr)

!
! Expand the eigenfunctions
!

      nr=npts
      do i=1,lh
         v(i,k2) = v2(i,kr)
         v(nr,k2)= v2(i,kr)
         nr=nr-1
      enddo

!
! Normalize the eigenfunction
!

      sn=0.d0
      do n=1,npts
         sn=sn+v(n,k2)*v(n,k2)
      enddo
      
      sn=1.d0/sqrt(sn)

!
! Put eigenfunctions positive standard
!

      if ((v(lh+1,k2).lt.0.d0)) then
         sn=-sn
      endif
      
      do n=1,npts
         v(n,k2)=sn*v(n,k2)
      enddo
  
   enddo

!
!  Do the odd tapers
!

   if (nodd > 0) then   

      if (allocated(v2)) then
         deallocate(v2)
      endif   
      allocate(v2(lh-nx,nodd))

      do i=1,lh
         n = i-1
         fv1(i)  = com*(hn - dble(n))**2
         fv2(i)  = dble(n*(npts-n))/2.d0
      enddo
   
      if (nx.eq.0) then
         fv1(lh)  =  com* (hn - dble(lh-1))**2 - dble(lh*(npts-lh))/2.d0
      endif
   
      call rst_eig(lh-nx,fv1,fv2(2:lh-nx),nodd,eigval,v2)
 
      do k = 1,nodd
         kr = k
         k2 = 2*k 

         theta(k2) = eigval(kr)

!
! Expand the eigenfunctions
!

         nr=npts
         do i=1,lh-nx
            v(i,k2) = v2(i,kr)
            v(nr,k2)= -v2(i,kr)
            nr=nr-1
         enddo
         if (nx == 1) then
            v(lh,k2) = 0.d0
         endif
   
!
! Normalize the eigenfunction
!

         sn=0.d0
         do n=1,npts
            sn=sn+v(n,k2)*v(n,k2)
         enddo
      
         sn=1.d0/sqrt(sn)

!
! Put eigenfunctions positive standard
!

         if ((nx==1) .and. (v(lh+1,k2).lt.0.d0)) then
            sn = -sn    
         endif
         if ((nx <= 0) .and. (v(lh+1,k2).lt.0.d0)) then
            sn=-sn
         endif

         do n=1,npts
            v(n,k2)=sn*v(n,k2)
         enddo

      enddo

   endif

   ntot=neven+nodd

!
!  Get the eigenvalues, by Quadrature (Chebychev)
!

   atol = 1.d-14

   call dpss_ev(npts,nev,bw,atol,v,lambda,theta)

   deallocate(fv1,fv2,v2)

   return
   
end subroutine dpss




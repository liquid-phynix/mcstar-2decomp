module iop

    use MPI
    use decomp_2d
    use iso_c_binding

    implicit none
    integer :: ierror

contains

    !subroutine initialize()
        !call MPI_INIT(ierror)
    !end subroutine

    !subroutine finalize()
        !call MPI_FINALIZE(ierror)
    !end subroutine

    subroutine get_float_size(s)
        integer (c_int), intent(out) :: s
        s = mytype_bytes
    end subroutine

    subroutine sync_configuration(out_xstart, out_ystart, out_zstart, &
    out_xend, out_yend, out_zend, out_xsize, out_ysize, out_zsize)

        integer (c_int), dimension(3), intent(out) :: out_xstart, &
            out_ystart, out_zstart, out_xend, out_yend, out_zend, &
            out_xsize, out_ysize, out_zsize

        out_xstart(:) = xstart(:) - 1
        out_ystart(:) = ystart(:) - 1
        out_zstart(:) = zstart(:) - 1
        out_xend  (:) = xend  (:) - 1
        out_yend  (:) = yend  (:) - 1
        out_zend  (:) = zend  (:) - 1
        out_xsize (:) = xsize (:)
        out_ysize (:) = ysize (:)
        out_zsize (:) = zsize (:)

    end subroutine

    !subroutine interop_decomp_2d_init(nx,ny,nz,prow,pcol)
    !integer, intent(in) :: nx, ny, nz, prow, pcol
    !!integer (c_int), value, intent(in) :: nx, ny, nz, prow, pcol
        !write (*,*) "in interop_decomp_2d_init"
        !write (*,*) nx, ny, nz, prow, pcol
        !call decomp_2d_init(nx, ny, nz, prow, pcol)
    !end subroutine

end module

  !implicit none

  !integer, parameter :: nx=17, ny=13, nz=11
  !integer, parameter :: p_row=4, p_col=3

  !real(mytype), dimension(nx,ny,nz) :: data1
  
  !real(mytype), allocatable, dimension(:,:,:) :: u1, u2, u3
  
  !integer :: i,j,k, m, ierror
  
  !call MPI_INIT(ierror)
  !call decomp_2d_init(nx,ny,nz,p_row,p_col)

  !! ***** global data *****
  !m = 1
  !do k=1,nz
     !do j=1,ny
        !do i=1,nx
           !data1(i,j,k) = float(m)
           !m = m+1
        !end do
     !end do
  !end do

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Testing the swap routines
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  !!allocate(u1(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)))
  !!allocate(u2(ystart(1):yend(1), ystart(2):yend(2), ystart(3):yend(3)))
  !!allocate(u3(zstart(1):zend(1), zstart(2):zend(2), zstart(3):zend(3)))
  !call alloc_x(u1, opt_global=.true.)
  !call alloc_y(u2, opt_global=.true.)
  !call alloc_z(u3, opt_global=.true.)
 
  !! original x-pensil based data 
  !do k=xstart(3),xend(3)
    !do j=xstart(2),xend(2)
      !do i=xstart(1),xend(1)
        !u1(i,j,k) = data1(i,j,k)
      !end do
    !end do
  !end do

!10 format(15I5)

  !if (nrank==0) then 
     !write(*,*) 'Numbers held on Rank 0'
     !write(*,*) ' '
     !write(*,*) 'X-pencil'
     !write(*,10) int(u1)
  !end if

  !call decomp_2d_write_one(1,u1,'u1.dat')

  !!!!!!!!!!!!!!!!!!!!!!!!
  !! x-pensil ==> y-pensil
  !call transpose_x_to_y(u1,u2)

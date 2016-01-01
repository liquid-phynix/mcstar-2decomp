MODULE decomp_2d_interop

 USE MPI
 USE decomp_2d
 USE decomp_2d_io
 USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_ptr, c_int, c_char, c_f_pointer

 IMPLICIT NONE
 TYPE(decomp_info), DIMENSION(:), ALLOCATABLE :: decomp_info_table

CONTAINS

 SUBROUTINE get_float_size(s) BIND(C)
  INTEGER (c_int), INTENT(OUT) :: s
  s = mytype_bytes
 END SUBROUTINE

 SUBROUTINE initialize(g_shape, proc_grid) BIND(C)
  INTEGER(c_int), DIMENSION(3) :: g_shape
  INTEGER(c_int), DIMENSION(2) :: proc_grid
  CALL decomp_2d_init(g_shape(1), g_shape(2), g_shape(3), proc_grid(1), proc_grid(2))
 END SUBROUTINE

 SUBROUTINE finalize() BIND(C)
  INTEGER :: i
  IF(ALLOCATED(decomp_info_table)) THEN
   DO i = 1, SIZE(decomp_info_table)
    CALL decomp_info_finalize(decomp_info_table(i))
   END DO
  END IF
  CALL decomp_2d_finalize()
 END SUBROUTINE

 SUBROUTINE extend_decomp_info_table(table, di)
  TYPE(decomp_info) :: di
  TYPE(decomp_info), DIMENSION(:), ALLOCATABLE :: table, table2
  IF(.NOT. ALLOCATED(table)) THEN
   ALLOCATE(table(1))
   table(1) = di
  ELSE
   ALLOCATE(table2(SIZE(table) + 1))
   table2(1:SIZE(table)) = table
   DEALLOCATE(table)
   table2(SIZE(table2)) = di
   CALL MOVE_ALLOC(table2, table)
  END IF
 END SUBROUTINE

 SUBROUTINE create_decomp_info(a_shape, table_index, rank, config) BIND(C)
  INTEGER(c_int), DIMENSION(3) :: a_shape
  INTEGER(c_int), INTENT(OUT) :: table_index, rank
  INTEGER(c_int), DIMENSION(3 * 3 * 3), INTENT(OUT) :: config
  TYPE(decomp_info) :: di
  CALL decomp_info_init(a_shape(1), a_shape(2), a_shape(3), di)
  CALL extend_decomp_info_table(decomp_info_table, di)
  rank = nrank
  table_index = SIZE(decomp_info_table)
  config = [di%xst-1, di%xen-1, di%xsz, di%yst-1, di%yen-1, di%ysz, di%zst-1, di%zen-1, di%zsz]
  !WRITE (*,*) 'fort: rank ', nrank, ' xsize ', di%xsz
  !WRITE (*,*) 'fort: rank ', nrank, ' ysize ', di%ysz
  !WRITE (*,*) 'fort: rank ', nrank, ' zsize ', di%zsz
 END SUBROUTINE

 SUBROUTINE save_array(array_ptr, elem_size, decomp_info_index, pencil_kind, fn, fn_len) BIND(C)
  INTEGER(c_int), VALUE :: elem_size, decomp_info_index, pencil_kind, fn_len
  TYPE(c_ptr), VALUE :: array_ptr
  CHARACTER(1, KIND=c_char), INTENT(IN) :: fn(fn_len)
  CHARACTER(fn_len, KIND=c_char) :: fnfort
  REAL(mytype), POINTER :: array_real(:,:,:)
  COMPLEX(mytype), POINTER :: array_cmpl(:,:,:)
  TYPE(decomp_info) :: di
  INTEGER :: i

  do i=1,size(fn)
   fnfort(i:i)=fn(i)
  end do

  di = decomp_info_table(decomp_info_index)
  SELECT CASE (elem_size / mytype_bytes)
  CASE (1) ! real elements
   SELECT CASE (pencil_kind)
   CASE (1)
    CALL c_f_pointer(array_ptr, array_real, di%xsz)
   CASE (2)
    CALL c_f_pointer(array_ptr, array_real, di%ysz)
   CASE (3)
    CALL c_f_pointer(array_ptr, array_real, di%zsz)
   CASE DEFAULT
    STOP "'save_array/real': cannot happen"
   END SELECT
  CALL decomp_2d_write_one(pencil_kind, array_real, fnfort, di)
  CASE (2) ! complex elements
   SELECT CASE (pencil_kind)
   CASE (1)
    CALL c_f_pointer(array_ptr, array_cmpl, di%xsz)
   CASE (2)
    CALL c_f_pointer(array_ptr, array_cmpl, di%ysz)
   CASE (3)
    CALL c_f_pointer(array_ptr, array_cmpl, di%zsz)
   CASE DEFAULT
    STOP "'save_array/cmpl': cannot happen"
   END SELECT
  CALL decomp_2d_write_one(pencil_kind, array_cmpl, fnfort, di)
  CASE DEFAULT
   STOP "'save_array/size': cannot happen"
  END SELECT
  IF (nrank .EQ. 0) THEN
      WRITE(*,*) 'written file <', fnfort, '>'
  END IF
 END SUBROUTINE

 SUBROUTINE global_transposition(in_ptr, in_pencil, out_ptr, out_pencil, decomp_info_index) BIND(C)
  TYPE(c_ptr), VALUE :: in_ptr, out_ptr
  INTEGER(c_int), VALUE :: in_pencil, out_pencil, decomp_info_index
  COMPLEX(mytype), POINTER :: array_in(:,:,:), array_out(:,:,:)
  TYPE(decomp_info) :: di
  di = decomp_info_table(decomp_info_index)
  SELECT CASE (in_pencil)
   CASE (1) ! from x
    CALL c_f_pointer(in_ptr, array_in, di%xsz)
    IF (out_pencil .EQ. 2) THEN ! to y
     CALL c_f_pointer(out_ptr, array_out, di%ysz)
     CALL transpose_x_to_y(array_in, array_out, di)
    ELSE
     WRITE(*,*) 'global_transposition: in-pencil: ', in_pencil, ', out-pencil: ', out_pencil
     STOP "wrong transposition"
    END IF
   CASE (2) ! from y
    CALL c_f_pointer(in_ptr, array_in, di%ysz)
    IF (out_pencil .EQ. 1) THEN ! to x
     CALL c_f_pointer(out_ptr, array_out, di%xsz)
     CALL transpose_y_to_x(array_in, array_out, di)
    ELSEIF (out_pencil .EQ. 3) THEN ! to z
     CALL c_f_pointer(out_ptr, array_out, di%zsz)
     CALL transpose_y_to_z(array_in, array_out, di)
    ELSE
     WRITE(*,*) 'global_transposition: in-pencil: ', in_pencil, ', out-pencil: ', out_pencil
     STOP "wrong transposition"
    END IF
   CASE (3) ! from z
    CALL c_f_pointer(in_ptr, array_in, di%zsz)
    IF (out_pencil .EQ. 2) THEN ! to y
     CALL c_f_pointer(out_ptr, array_out, di%ysz)
     CALL transpose_z_to_y(array_in, array_out, di)
    ELSE
     WRITE(*,*) 'global_transposition: in-pencil: ', in_pencil, ', out-pencil: ', out_pencil
     STOP "wrong transposition"
    END IF
   CASE DEFAULT
    STOP "'global_transposition': cannot happen"
  END SELECT
 END SUBROUTINE

END MODULE

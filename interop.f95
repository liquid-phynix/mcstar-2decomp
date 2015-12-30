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
 END SUBROUTINE

 SUBROUTINE save_array(array_ptr, elem_size, decomp_info_index, pencil_kind, fn_ptr, fn_len) BIND(C)
  INTEGER(c_int), INTENT(IN) :: elem_size, decomp_info_index, pencil_kind, fn_len
  TYPE(c_ptr) :: array_ptr, fn_ptr
  REAL(mytype), POINTER :: array_real(:,:,:)
  COMPLEX(mytype), POINTER :: array_cmpl(:,:,:)
  CHARACTER(KIND=c_char, LEN=fn_len), POINTER :: fn
  TYPE(decomp_info) :: di
  di = decomp_info_table(decomp_info_index)
  CALL c_f_pointer(fn_ptr, fn)
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
    STOP "cannot happen"
   END SELECT
  CALL decomp_2d_write_one(pencil_kind, array_real, fn, di)
  CASE (2) ! complex elements
   SELECT CASE (pencil_kind)
   CASE (1)
    CALL c_f_pointer(array_ptr, array_cmpl, di%xsz)
   CASE (2)
    CALL c_f_pointer(array_ptr, array_cmpl, di%ysz)
   CASE (3)
    CALL c_f_pointer(array_ptr, array_cmpl, di%zsz)
   CASE DEFAULT
    STOP "cannot happen"
   END SELECT
  CALL decomp_2d_write_one(pencil_kind, array_cmpl, fn, di)
  CASE DEFAULT
   STOP "cannot happen"
  END SELECT
  IF (nrank .EQ. 0) THEN
      WRITE(*,*) 'written file <', fn, '>'
  END IF
 END SUBROUTINE


    !subroutine iop_transpose(ttype, input_ptr, input_sizes, output_ptr, output_sizes)
        !! ttype = (1: x->y) | (2: y->z) | (3: z->y) | (4: y->x)
        !integer (c_int), intent(in) :: ttype
        !integer (c_int), dimension(3), intent(in) :: input_sizes, output_sizes
        !type (c_ptr)                :: input_ptr, output_ptr
        !complex(mytype), pointer    :: input(:,:,:), output(:,:,:)

        !call c_f_pointer(input_ptr, input, input_sizes)
        !call c_f_pointer(output_ptr, output, output_sizes)

        !select case (ttype)
        !case (1)
            !call transpose_x_to_y(input, output)
        !case (2)
            !call transpose_y_to_z(input, output)
        !case (3)
            !call transpose_z_to_y(input, output)
        !case (4)
            !call transpose_y_to_x(input, output)
        !case default
            !stop "ttype not in [1,4]"
        !end select

    !end subroutine

end module

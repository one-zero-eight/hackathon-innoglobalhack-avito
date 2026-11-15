import { brandAssets } from '../theme'

interface AvitoLogoProps {
  /** Высота логотипа в px (ширина считается по пропорциям оригинала). */
  height?: number
  /** Альтернативный текст; для декоративных мест передавайте пустую строку. */
  alt?: string
  /** true — только знак из четырёх точек, без вордмарка. */
  markOnly?: boolean
  className?: string
}

/** Пропорции официальных SVG: вордмарк 111×30, знак 30×30. */
const LOGO_RATIO = 111 / 30
const MARK_RATIO = 1

/**
 * Официальный логотип Avito из `public/`: знак из четырёх цветных точек
 * и вордмарк. `markOnly` оставляет только точки — для скелетонов загрузки,
 * ватермарка и фавиконки. Обычный `<img>` вместо Mantine `Image`, чтобы
 * размер не переопределялся правилом `width: 100%` из стилей компонента.
 */
export function AvitoLogo({ height = 30, alt = 'Avito', markOnly = false, className }: AvitoLogoProps) {
  return (
    <img
      src={markOnly ? brandAssets.mark : brandAssets.logo}
      alt={alt}
      width={Math.round(height * (markOnly ? MARK_RATIO : LOGO_RATIO))}
      height={height}
      draggable={false}
      className={className}
      style={{ display: 'block' }}
    />
  )
}

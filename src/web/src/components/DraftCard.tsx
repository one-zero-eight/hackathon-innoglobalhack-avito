import {
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { Badge, Box, Button, Card, Group, Stack, Text } from '@mantine/core'
import { accentColor, badgeColors, microcategoryPalette } from '../theme'
import type { Draft, Microcategory } from '../api'

const COLLAPSED_LINE_COUNT = 6
const STEM_LENGTH = 5
const WORD_SOURCE = String.raw`[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*`
const WORDS_RE = new RegExp(WORD_SOURCE, 'gu')
const WORD_PARTS_RE = new RegExp(`(${WORD_SOURCE})`, 'gu')
const SINGLE_WORD_RE = new RegExp(`^${WORD_SOURCE}$`, 'u')
const WORD_ENDING_RE =
  /(?:ические|ического|ическому|ическим|ических|ический|ическая|ическое|иями|ями|ами|ские|ская|ского|ских|ский|ого|его|ому|ему|ыми|ими|ов|ев|ых|их|ам|ям|ах|ях|ый|ий|ой|ая|яя|ое|ее|ые|ие|ым|им|ом|ем|ть|ут|ют|ат|ят|ла|ло|ли|а|я|ы|и|е|о|у|ю|ет|ит)$/u
const STOP_WORDS = new Set([
  'а',
  'без',
  'в',
  'для',
  'и',
  'из',
  'к',
  'на',
  'о',
  'от',
  'по',
  'под',
  'при',
  'с',
  'у',
  'ключ',
  'работа',
  'работы',
  'услуга',
  'услуги',
])

interface DraftCardProps {
  draft: Draft
  /** Список для сопоставления mcId → названия микрокатегории. */
  microcategories: Microcategory[]
  /** Исходная mcId объявления — подсветка нежелательного совпадения. */
  sourceMcId: number
}

interface DraftTextProps {
  text: string
  keywordSources: string[]
}

function titleFor(microcategories: Microcategory[], mcId: number): string {
  const found = microcategories.find((item) => item.mcId === mcId)
  return found?.title ?? `Микрокатегория ${mcId}`
}

function normalizeWord(word: string): string {
  return word.toLocaleLowerCase('ru').replaceAll('ё', 'е')
}

function wordStem(word: string): string | null {
  const normalized = normalizeWord(word)
  if (normalized.length < 3 || STOP_WORDS.has(normalized)) {
    return null
  }

  const withoutEnding = normalized.replace(WORD_ENDING_RE, '')
  const stem = withoutEnding.length >= 3 ? withoutEnding : normalized
  return stem.slice(0, STEM_LENGTH)
}

function collectStems(sources: string[]): Set<string> {
  const stems = new Set<string>()
  for (const source of sources) {
    for (const match of source.matchAll(WORDS_RE)) {
      const stem = wordStem(match[0])
      if (stem !== null) {
        stems.add(stem)
      }
    }
  }
  return stems
}

function highlightKeywords(text: string, keywordSources: string[]): ReactNode[] {
  const stems = collectStems(keywordSources)
  return text.split(WORD_PARTS_RE).map((part, index) => {
    if (!SINGLE_WORD_RE.test(part)) {
      return part
    }

    const stem = wordStem(part)
    return stem !== null && stems.has(stem) ? <strong key={index}>{part}</strong> : part
  })
}

/** Текст черновика с подсветкой услуг и раскрытием только при переполнении. */
function DraftText({ text, keywordSources }: DraftTextProps) {
  const [expanded, setExpanded] = useState(false)
  const [overflowing, setOverflowing] = useState(false)
  const textRef = useRef<HTMLParagraphElement>(null)
  const textId = useId()

  const content = useMemo(() => highlightKeywords(text, keywordSources), [text, keywordSources])

  useLayoutEffect(() => {
    const element = textRef.current
    if (element === null || expanded) {
      return
    }

    const measure = () => {
      setOverflowing(element.scrollHeight > element.clientHeight + 1)
    }
    measure()

    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [expanded, text])

  return (
    <Stack gap={0}>
      <Box pos="relative">
        <Text
          id={textId}
          ref={textRef}
          size="sm"
          c="dark.7"
          lh={1.5}
          lineClamp={expanded ? undefined : COLLAPSED_LINE_COUNT}
          style={{ whiteSpace: 'pre-line' }}
          className="break-words hyphens-auto"
        >
          {content}
        </Text>
        {!expanded && overflowing && (
          <Box
            aria-hidden
            pos="absolute"
            bottom={0}
            left={0}
            right={0}
            h={42}
            style={{
              background: 'linear-gradient(to bottom, transparent, var(--mantine-color-white))',
              pointerEvents: 'none',
            }}
          />
        )}
      </Box>

      {(overflowing || expanded) && (
        <Button
          type="button"
          variant="subtle"
          size="sm"
          w="fit-content"
          px={0}
          aria-expanded={expanded}
          aria-controls={textId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? 'Скрыть' : 'Показать полностью'}
        </Button>
      )}
    </Stack>
  )
}

/**
 * Карточка одного черновика: цветная топ-полоска и бейдж микрокатегории
 * в фирменной паре цветов, заголовок и текст объявления.
 */
export function DraftCard({ draft, microcategories, sourceMcId }: DraftCardProps) {
  const notGenerated = draft.title === null || draft.text === null
  const sameAsSource = draft.mcId === sourceMcId
  const categoryTitle = titleFor(microcategories, draft.mcId)
  const keywordSources = useMemo(
    () => [categoryTitle, draft.title ?? ''],
    [categoryTitle, draft.title],
  )

  // Совпадение с исходной категорией — ошибка подбора, подсвечиваем красным.
  const palette = sameAsSource ? null : microcategoryPalette(draft.mcId)
  const badge = palette === null ? badgeColors('brandCoral') : badgeColors(palette)
  const strip = palette === null ? accentColor('brandCoral') : accentColor(palette)

  return (
    <Card
      withBorder
      shadow="xs"
      padding={0}
      radius="md"
      bg="white"
      className="avito-card"
      style={{ overflow: 'hidden' }}
    >
      <Box aria-hidden className="avito-card-accent" style={{ background: strip }} />

      <Stack gap="sm" p="md">
        <Group gap="xs">
          <Badge
            radius="md"
            size="md"
            styles={{
              root: { backgroundColor: badge.bg, color: badge.text },
            }}
          >
            {draft.mcId} — {categoryTitle}
          </Badge>
        </Group>

        {draft.title !== null && (
          <Text fw={600} size="lg" lh={1.3} className="text-balance">
            {draft.title}
          </Text>
        )}

        {draft.text !== null && <DraftText text={draft.text} keywordSources={keywordSources} />}

        {notGenerated && (
          <Badge color="gray" variant="outline" radius="md" w="fit-content">
            Текст не сгенерирован
          </Badge>
        )}

        {sameAsSource && (
          <Text size="xs" c="brandCoral.8">
            Совпало с исходной микрокатегорией — проверьте подбор.
          </Text>
        )}
      </Stack>
    </Card>
  )
}
